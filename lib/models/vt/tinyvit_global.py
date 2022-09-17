# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
from collections import OrderedDict


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution_x,resolution_z, activation):
        super().__init__()
        search_size: Tuple[int, int] = to_2tuple(resolution_x)
        template_size: Tuple[int, int] = to_2tuple(resolution_z)
        self.patches_resolution_x = (search_size[0] // 4, search_size[1] // 4)
        self.patches_resolution_z = (template_size[0] // 4, template_size[1] // 4)
        self.num_patches_x = self.patches_resolution_x[0] * \
            self.patches_resolution_x[1]
        self.num_patches_z = self.patches_resolution_z[0] * \
            self.patches_resolution_z[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution_x,input_resolution_z, dim,
                 out_dim, activation,template_number):
        super().__init__()

        self.input_resolution_x = input_resolution_x
        self.input_resolution_z = input_resolution_z
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)
        self.num = template_number + 1
    def forward(self, x):
        if x.ndim == 3:  # (B,L,C)
            H_x, W_x = self.input_resolution_x
            H_z, W_z = self.input_resolution_z
            B = len(x)
            for i in range(self.num):
                if i == 0:
                    H = H_x
                    W = W_x
                else:
                    H = H_z
                    W = W_z
                xm = x[:,H*W*i:H*W*(i+1):].transpose(1,2).view(B,-1,H,W)
                xm = self.conv1(xm)
                xm = self.act(xm)
                xm = self.conv2(xm)
                xm = self.act(xm)
                xm = self.conv3(xm)
                xm = xm.flatten(2).transpose(1, 2)
                if i == 0:
                    xz = xm
                else:
                    xz = torch.cat((xz,xm),dim=1)
            x = xz
        else:

            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

            x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution_x,input_resolution_z, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 template_number=1,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution_x = input_resolution_x
        self.input_resolution_z = input_resolution_z
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.template_number = template_number

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution_x,input_resolution_z, dim=dim, out_dim=out_dim, activation=activation,template_number=template_number)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 resolution_x = (256,256),
                 resolution_z = (256,256),
                 template_number = 1,
                 global_att = False,
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)
        if global_att == False:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(num_heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                 torch.LongTensor(idxs).view(N, N),
                                 persistent=False)
        else:
            points = list(itertools.product(
                range(resolution_x[0]), range(resolution_x[1])))
            for i in range(template_number):
                points += list(itertools.product(range(resolution_x[0]+resolution_z[0]*i,resolution_x[1]+resolution_z[1]*(i+1)),
                                                 range(resolution_x[0]+resolution_z[0]*i,resolution_x[1]+resolution_z[1]*(i+1))))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(num_heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                 torch.LongTensor(idxs).view(N, N),
                                 persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def  __init__(self, dim, input_resolution_x,input_resolution_z, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 num_depth=0,
                 total_depth=2,
                 template_number=1
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution_x = input_resolution_x
        self.input_resolution_z = input_resolution_z
        H_x, W_x = self.input_resolution_x
        H_z, W_z = self.input_resolution_z
        L = H_x * W_x + H_z * W_z
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_depth = num_depth
        self.total_depth = total_depth
        self.template_number = template_number

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        # window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                                attn_ratio=1, resolution=(L, L),
                                resolution_x=self.input_resolution_x,
                                resolution_z=self.input_resolution_z,
                                template_number=self.template_number,
                                global_att=True,
                                )
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, xz,res_out): #(B,HW,C)
        H_x, W_x = self.input_resolution_x
        H_z, W_z = self.input_resolution_z
        B = len(xz)
        xm = self.attn(xz)  # (b/N,L*N,C)
        xm = xz + self.drop_path(xm)
        for i in range(self.template_number+1):
            if i == 0:
                H = H_x
                W = W_x
            else:
                H = H_z
                W = W_z
            xc = xm[:,H*W*i:H*W*(i+1),:].transpose(1,2).reshape(B,-1,H,W)
            xc = self.local_conv(xc)
            xc = xc.view(B,-1,H*W).transpose(1,2)
            if i == 0:
                xcm = xc
            else:
                xcm = torch.cat((xcm,xc),dim=1)
        xm = xcm
        xm = xm + self.drop_path(self.mlp(xm))
        if self.num_depth == self.total_depth - 1:
            for i in range(self.template_number+1):
                if i == 0:
                    res_out.append(xm[:, H_x*W_x*i:H_x*W_x*(i+1), :])

        return xm,res_out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution_x={self.input_resolution_x}, input_resolution_z={self.input_resolution_z},num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def  __init__(self, dim, input_resolution_x,input_resolution_z, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 template_number=1,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution_x = input_resolution_x
        self.input_resolution_z = input_resolution_z
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.template_number = template_number
        # # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution_x=input_resolution_x,
                         input_resolution_z=input_resolution_z,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         num_depth=i,
                         total_depth=depth,
                         template_number=self.template_number,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution_x,input_resolution_z, dim=dim, out_dim=out_dim, activation=activation,template_number=template_number)
        else:
            self.downsample = None

    def forward(self, x,res_x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x,res_x = checkpoint.checkpoint(blk, x,res_x)
            else:
                x,res_x = blk(x,res_x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x,res_x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution_x={self.input_resolution_x}, input_resolution_z={self.input_resolution_z},depth={self.depth}"


class TinyViT(nn.Module):
    def  __init__(self, search_size=256,template_size=128,
                 template_number=1,neck_type='LINEAR',
                 in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()


        self.embed_dim_list = embed_dims
        self.neck_type = neck_type
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.template_number = template_number
        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution_x=search_size,
                                      resolution_z=template_size,
                                      activation=activation)

        patches_resolution_x = self.patch_embed.patches_resolution_x
        patches_resolution_z = self.patch_embed.patches_resolution_z
        self.patches_resolution_x = patches_resolution_x    #(H,W)
        self.patches_resolution_z = patches_resolution_z
        self.num_patches_search = (patches_resolution_x[0] //(2**(self.num_layers-1))) ** 2
        self.num_patches_template = (patches_resolution_z[0] //(2**(self.num_layers-1))) ** 2

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()


        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution_x=(patches_resolution_x[0] // (2 ** i_layer),
                                            patches_resolution_x[1] // (2 ** i_layer)),
                          input_resolution_z=(patches_resolution_z[0] // (2 ** i_layer),
                                              patches_resolution_z[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          template_number = self.template_number
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)
        # Classifier head


        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        # for m in [self.norm_head, self.head]:
        #     m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, image_lists):
        # x: (N, C, H, W)
        res_out = []
        for i in range(len(image_lists)):
            x = image_lists[i]
            x = self.patch_embed(x)
            x = self.layers[0](x)#(B,L,C)
            if i == 0:
                xz = x
            else:
                xz = torch.cat((xz,x),dim=1)
        start_i = 1


        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            xz,res_out = layer(xz,res_out) #out==image_list,res_out==list
        # out[i] == [B,L,C]
        cls = xz.mean(1).unsqueeze(1)
        # cls = res_out[-1].mean(1).unsqueeze(1)#[bs,1,576]
        res_out.append(cls)#[stage2_out[0],stage3_out[0],stage4_out[0],cls]

        # x = x.mean(1)

        return res_out

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x


_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'
_provided_checkpoints = {
    'tiny_vit_5m_224': 'tiny_vit_5m_22kto1k_distill',
    'tiny_vit_11m_224': 'tiny_vit_11m_22kto1k_distill',
    'tiny_vit_21m_224': 'tiny_vit_21m_22kto1k_distill',
    'tiny_vit_21m_384': 'tiny_vit_21m_22kto1k_384_distill',
    'tiny_vit_21m_512': 'tiny_vit_21m_22kto1k_512_distill',
}


def load_pretrained(model,model_name):
    assert model_name in _provided_checkpoints, \
        f'Sorry that the checkpoint `{model_name}` is not provided yet.'
    url = _checkpoint_url_format.format(
        _provided_checkpoints[model_name])
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu', check_hash=False,
    )
    state_dict = checkpoint['model']
    state_dict_load = OrderedDict()
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            if ("attention_bias" not in key):
                state_dict_load[key] = state_dict[key]#attention bias与图片大小有关不能直接load
            else:
                state_dict_load[key] = model.state_dict()[key]
    model.load_state_dict(state_dict_load)


def tiny_vit_5m_224(pretrained=False, num_classes=1000, drop_path_rate=0.0):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )



def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )



def tiny_vit_21m_224(pretrained=False, num_classes=1000,
                     drop_path_rate=0.2,search_size=256,template_size=128,
                     template_number=1,neck_type="FPN"):
    model = TinyViT(
        search_size=search_size,
        template_size=template_size,
        template_number=template_number,
        neck_type=neck_type,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )
    if pretrained:
        model_name =  "tiny_vit_21m_224"
        load_pretrained(model, model_name)

    return model



def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=384,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
    )



def tiny_vit_21m_512(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
    )