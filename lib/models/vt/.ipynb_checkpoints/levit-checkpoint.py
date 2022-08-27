# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License
from collections import OrderedDict

import torch
import itertools
import lib.models.vt.levit_utils as utils

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model

specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'LeViT_128': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'LeViT_192': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'LeViT_256': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'LeViT_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

__all__ = [specification.keys()]


# @register_model
# def LeViT_128S(num_classes=1000, distillation=True,
#                pretrained=False, fuse=False):
#     return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
#                          distillation=distillation, pretrained=pretrained, fuse=fuse)
#
#
# @register_model
# def LeViT_128(num_classes=1000, distillation=True,
#               pretrained=False, fuse=False):
#     return model_factory(**specification['LeViT_128'], num_classes=num_classes,
#                          distillation=distillation, pretrained=pretrained, fuse=fuse)
#
#
# @register_model
# def LeViT_192(num_classes=1000, distillation=True,
#               pretrained=False, fuse=False):
#     return model_factory(**specification['LeViT_192'], num_classes=num_classes,
#                          distillation=distillation, pretrained=pretrained, fuse=fuse)
#
#
# @register_model
# def LeViT_256(num_classes=1000, distillation=True,
#               pretrained=False, fuse=False):
#     return model_factory(**specification['LeViT_256'], num_classes=num_classes,
#                          distillation=distillation, pretrained=pretrained, fuse=fuse)
#
#
# @register_model
# def LeViT_384(num_classes=1000, distillation=True,
#               pretrained=False, fuse=False):
#     return model_factory(**specification['LeViT_384'], num_classes=num_classes,
#                          distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def LeViT_128S(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              search_size=224, template_size=112, template_number=1, neck_type='LINEAR'):
    return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         search_size=search_size, template_size=template_size, template_number=template_number,
                         neck_type=neck_type)


@register_model
def LeViT_128(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              search_size=224, template_size=112, template_number=1, neck_type='LINEAR'):
    return model_factory(**specification['LeViT_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         search_size=search_size, template_size=template_size, template_number=template_number,
                         neck_type=neck_type)


@register_model
def LeViT_192(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              search_size=224, template_size=112, template_number=1, neck_type='LINEAR'):
    return model_factory(**specification['LeViT_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         search_size=search_size, template_size=template_size, template_number=template_number,
                         neck_type=neck_type)


@register_model
def LeViT_256(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              search_size=224, template_size=112, template_number=1, neck_type='LINEAR'):
    return model_factory(**specification['LeViT_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         search_size=search_size, template_size=template_size, template_number=template_number,
                         neck_type=neck_type)


@register_model
def LeViT_384(num_classes=1000, distillation=True,
              pretrained=False, fuse=False,
              search_size=224, template_size=112, template_number=1, neck_type='LINEAR'):
    return model_factory(**specification['LeViT_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         search_size=search_size, template_size=template_size, template_number=template_number,
                         neck_type=neck_type)

FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

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


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution_x=-100000, resolution_z=-100000, template_number=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution_x**2 + (resolution_z**2)*template_number
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution_x=16,
                 resolution_z=8,
                 template_number=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution_x=resolution_x, resolution_z=resolution_z, template_number=template_number)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution_x=resolution_x, resolution_z=resolution_z, template_number=template_number))


        points = list(itertools.product(range(resolution_x), range(resolution_x)))
        for i in range(template_number):
            points += list(itertools.product(range(resolution_x+resolution_z*i,resolution_x+resolution_z*(i+1)), range(resolution_x+resolution_z*i,resolution_x+resolution_z*(i+1))))
        N_xz = len(points)
        attention_offsets_xz = {}
        idxs_xz = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets_xz:
                    attention_offsets_xz[offset] = len(attention_offsets_xz)
                idxs_xz.append(attention_offsets_xz[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets_xz)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs_xz).view(N_xz, N_xz))
        # N_xz = resolution_x**2 + (resolution_z**2)*template_number
        # idxs_xz = []
        # for i in range(N_xz):
        #     for j in range(N_xz):
        #         idxs_xz.append(abs(j-i))
        # self.attention_biases = torch.nn.Parameter(
        #     torch.zeros(num_heads, N_xz))
        # self.register_buffer('attention_bias_idxs',
        #                      torch.LongTensor(idxs_xz).view(N_xz, N_xz))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (N_xz**2) * key_dim
        # # softmax
        FLOPS_COUNTER += num_heads * (N_xz**2)
        # #attention * v
        FLOPS_COUNTER += num_heads * self.d * (N_xz**2)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
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

class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution_x, resolution_z, template_number):
        super().__init__()
        self.stride = stride
        self.resolution_x = resolution_x
        self.resolution_z = resolution_z
        self.template_number = template_number

    def forward(self, xz):
        B, N, C = xz.shape
        for i in range(self.template_number+1):
            if i == 0:
                xz_ = xz[:,0:self.resolution_x**2,:]
                xz_ = xz_.view(B, self.resolution_x, self.resolution_x, C)[
                    :, ::self.stride, ::self.stride].reshape(B, -1, C)
                xz_ = xz_
            else:
                z = xz[:,(self.resolution_x**2+(i-1)*(self.resolution_z**2)):(self.resolution_x**2+i*(self.resolution_z**2)),:]
                z = z.view(B, self.resolution_z, self.resolution_z, C)[
                    :, ::self.stride, ::self.stride].reshape(B, -1, C)
                xz_ = torch.cat((xz_, z), dim=1)

        # x = x.view(B, self.resolution, self.resolution, C)[
        #     :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return xz_

class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution_x=16,
                 resolution_z=8,
                 resolution_x_=8,
                 resolution_z_=4,
                 template_number=1):

        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_x_ = resolution_x_
        self.resolution_z_ = resolution_z_
        self.resolution_x_2 = resolution_x_**2
        self.resolution_z_2 = resolution_z_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution_x=resolution_x, resolution_z=resolution_z, template_number=template_number)
        self.q = torch.nn.Sequential(
            Subsample(stride, resolution_x, resolution_z, template_number),
            Linear_BN(in_dim, nh_kd, resolution_x=resolution_x_, resolution_z=resolution_z_, template_number=template_number))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution_x=resolution_x_, resolution_z=resolution_z_, template_number=template_number))

        self.stride = stride
        self.resolution_x = resolution_x
        self.resolution_z = resolution_z
        points = list(itertools.product(range(resolution_x), range(resolution_x)))
        for i in range(template_number):
            points += list(
                itertools.product(
                    range(resolution_x+resolution_z*i,resolution_x+resolution_z*(i+1)),
                    range(resolution_x+resolution_z*i,resolution_x+resolution_z*(i+1))))
        points_ = list(itertools.product(range(resolution_x_), range(resolution_x_)))
        for i in range(template_number):
            points_ += list(
                itertools.product(
                    range(resolution_x_+resolution_z_*i,resolution_x_+resolution_z_*(i+1)),
                    range(resolution_x_+resolution_z_*i,resolution_x_+resolution_z_*(i+1))))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1  #why there is a size
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
                         (resolution_x**2+(resolution_z**2)*template_number) * \
                         (resolution_x_**2+(resolution_z_**2)*template_number) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution_x**2+(resolution_z**2)*template_number) * \
                         (resolution_x_**2+(resolution_z_**2)*template_number)
        #attention * v
        FLOPS_COUNTER += num_heads * \
                         (resolution_x**2+(resolution_z**2)*template_number) * \
                         (resolution_x_**2+(resolution_z_**2)*template_number) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        # q = self.q(x).view(B, self.resolution_2, self.num_heads,
        #                    self.key_dim).permute(0, 2, 1, 3)
        q = self.q(x).view(B, -1, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 search_size=256,
                 template_size=128,
                 template_number=1,
                 patch_size=16,
                 neck_type='LINEAR',
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()
        global FLOPS_COUNTER

        self.neck_type = neck_type
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim_list = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.template_number = template_number
        self.blocks = []
        down_ops.append([''])
        resolution_x = search_size // patch_size
        resolution_z = template_size // patch_size

        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution_x=resolution_x,
                        resolution_z=resolution_z,
                        template_number=template_number
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution_x=resolution_x, resolution_z=resolution_z, template_number=template_number),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution_x=resolution_x, resolution_z=resolution_z, template_number=template_number),
                        ), drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_x_ = (resolution_x - 1) // do[5] + 1
                resolution_z_ = (resolution_z - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution_x=resolution_x,
                        resolution_z=resolution_z,
                        resolution_x_=resolution_x_,
                        resolution_z_=resolution_z_,
                        template_number=template_number
                    ))
                resolution_x = resolution_x_
                resolution_z = resolution_z_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution_x=resolution_x,
                                      resolution_z=resolution_z),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0,
                                resolution_x=resolution_x,
                                resolution_z=resolution_z),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.num_patches_search = resolution_x ** 2
        self.num_patches_template = resolution_z ** 2

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if self.neck_type == 'FPN' or self.neck_type == 'MAXF' or self.neck_type == "MAXMINF" or self.neck_type == "MAXMIDF" or self.neck_type == "MINMIDF" or self.neck_type == 'MIDF':
            fpn_idx = []
            for i in range(len(self.blocks)):
                if self.blocks[i]._get_name() == 'AttentionSubsample':
                    fpn_idx.append(i)
            self.fpn_idx = fpn_idx

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward_features(self, images_list):
        # x = images_list[0]
        # x = self.patch_embed(x)  # [bs, 3, 256, 256] --> [bs, 384, 16, 16]
        # x = x.flatten(2).transpose(1, 2) #[bs, 256, 384]
        for i in range(len(images_list)):
            x = images_list[i]
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            if i == 0:
                xz = x
            else:
                xz = torch.cat((xz, x), dim=1)
        if self.neck_type == 'FPN' or self.neck_type == "MAXF" or self.neck_type == "MAXMINF" or self.neck_type == "MAXMIDF" or self.neck_type == "MINMIDF" or self.neck_type == 'MIDF':
            assert len(self.fpn_idx) == 2
            xz1 = self.blocks[0:self.fpn_idx[0]](xz)
            xz2 = self.blocks[self.fpn_idx[0]:self.fpn_idx[1]](xz1)
            xz = self.blocks[self.fpn_idx[1]:](xz2)
            out_list = [xz1, xz2]
        else:
            xz = self.blocks(xz) #[bs, 20, 768]
            out_list = []

        cls = xz.mean(1).unsqueeze(1) #[bs, 1, 768]
        cxz = torch.cat((cls, xz), dim=1)
        # if self.distillation:
        #     x = self.head(x), self.head_dist(x)
        #     if not self.training:
        #         x = (x[0] + x[1]) / 2
        # else:
        #     x = self.head(x)
        out_list.append(cxz)
        return out_list

    def forward(self, images_list):
        out_list = self.forward_features(images_list)
        # x = self.head(x)
        # out=[xz]
        return out_list


def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse,
                  search_size, template_size, template_number,
                  neck_type):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation,
        search_size=search_size,
        template_size=template_size,
        template_number=template_number,
        neck_type=neck_type
    )
    # modify pretrained for debug by chenxin
    if pretrained:
        load_pretrained(model, weights)

    if fuse:
        # merge conv+bn to one operator, should be False when Training and be True when Evaluate for accelerate
        utils.replace_batchnorm(model)
    return model

def load_pretrained(model, weights):
    checkpoint = torch.hub.load_state_dict_from_url(
        weights, map_location='cpu')
    state_dict = checkpoint['model']
    # state_dict_load = OrderedDict()
    # for key in state_dict.keys():
    #     if key[0:7] == 'visual.':
    #         state_dict_load[key[7:]] = state_dict[key]
    state_dict_load = OrderedDict()
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            if ("attention_bias" not in key):
                state_dict_load[key] = state_dict[key]#attention bias与图片大小有关不能直接load
            else:
                state_dict_load[key] = model.state_dict()[key]
    model.load_state_dict(state_dict_load)


if __name__ == '__main__':
    for name in specification:
        net = globals()[name](fuse=True, pretrained=True)
        net.eval()
        net(torch.randn(4, 3, 224, 224))
        print(name,
              net.FLOPS, 'FLOPs',
              sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
