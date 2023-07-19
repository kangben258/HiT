import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
   'pvt_small' , 'pvt_large'
]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, search_H, search_W, template_H, template_W):#x==>(b,n,c)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_search = x[:,:search_W*search_H,:].permute(0, 2, 1).reshape(B, C, search_H, search_W)
            x_search = self.sr(x_search).reshape(B, C, -1).permute(0, 2, 1)
            x_template = x[:,search_W*search_H:,:].permute(0, 2, 1).reshape(B, C, template_H, template_W)
            x_template = self.sr(x_template).reshape(B, C, -1).permute(0, 2, 1)
            x_search = self.norm(x_search)
            x_template = self.norm(x_template)
            x_ = torch.cat((x_search,x_template),dim=1)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, search_H, search_W, template_H, template_W):
        x = x + self.drop_path(self.attn(self.norm1(x), search_H, search_W, template_H, template_W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, search_size=256, template_size=128,patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        search_size = to_2tuple(search_size)
        template_size = to_2tuple(template_size)
        patch_size = to_2tuple(patch_size)

        self.search_size = search_size
        self.template_size = template_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.search_H, self.search_W = search_size[0] // patch_size[0], search_size[1] // patch_size[1]
        self.template_H,self.template_W = template_size[0] // patch_size[0], template_size[1] // patch_size[1]
        self.num_search_patches = self.search_H * self.search_W
        self.num_template_patches = self.template_H * self.template_W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, iamge_list):
        for i in range(len(iamge_list)):
            x = iamge_list[i]
            B, C, H, W = x.shape

            x = self.proj(x).flatten(2).transpose(1, 2)
            x = self.norm(x)
            if i == 0:
                search_H, search_W = H // self.patch_size[0], W // self.patch_size[1]
                xz = x
            else:
                template_H, template_W = H // self.patch_size[0], W // self.patch_size[1]
                xz = torch.cat((xz,x),dim=1)


        return xz, (search_H, search_W,template_H,template_W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], search_size=256,template_size=128,
              template_number=1,neck_type="FB",num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.num_patches_search = (search_size//32)**2
        self.num_patches_template = (template_size//32)**2
        self.embed_dim_list = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(search_size=search_size if i == 0 else search_size // (2 ** (i + 1)),
                                     template_size=template_size if i==0 else template_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_search_patches = patch_embed.num_search_patches
            num_template_patches = patch_embed.num_template_patches
            num_patches = num_template_patches + num_search_patches

            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, search_H, search_W, template_H, template_W):
        if search_H * search_W == self.patch_embed1.num_search_patches and template_W * template_H == self.patch_embed1.num_template_patches:
            return pos_embed
        else:
            pos_search_embed = F.interpolate(
                pos_embed[:, :patch_embed.num_search_patches, :].reshape(1, patch_embed.search_H, patch_embed.search_W,
                                                                         -1).permute(0, 3, 1, 2),size=(search_H,search_W),
                mode="bilinear"
            ).reshape(1,-1,search_H*search_W).permute(0,2,1)
            pos_template_embed = F.interpolate(
                pos_embed[:, patch_embed.num_search_patches:, :].reshape(1, patch_embed.template_H, patch_embed.template_W,
                                                                         -1).permute(0, 3, 1, 2),
                size=(template_H, template_W),
                mode="bilinear"
            ).reshape(1, -1, template_H * template_W).permute(0, 2, 1)
            pos_embed = torch.cat((pos_search_embed,pos_template_embed),dim=1)
            return pos_embed

    def forward_features(self, x):#x==>image_list
        B = x[0].shape[0]
        res_out = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")

            x, (search_H, search_W, template_H, template_W) = patch_embed(x)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, search_H, search_W, template_H, template_W)#x==>(B,n,c)
            res_out.append(x[:, :search_W * search_H, :])
            if i != self.num_stages - 1:
                x_search = x[:,:search_W*search_H,:].reshape(B, search_H, search_W, -1).permute(0, 3, 1, 2).contiguous()
                x_template = x[:, search_W * search_H:, :].reshape(B, template_H, template_W, -1).permute(0, 3, 1, 2).contiguous()
                x = [x_search,x_template]
            else:
                x = self.norm(x)
                res_out[-1] = x[:, :search_W * search_H, :]
        cls = x.mean(1).unsqueeze(1)
        res_out.append(cls)

        return res_out

    def forward(self, x):#x==>image_list
        x = self.forward_features(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def load_pretrained(model,url):


    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu', check_hash=False,
    )
    state_dict = checkpoint
    state_dict_load = OrderedDict()
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            if ("pos_embed" not in key):
                state_dict_load[key] = state_dict[key]
            else:
                state_dict_load[key] = model.state_dict()[key]
    model.load_state_dict(state_dict_load)


@register_model
def pvit_small(pretrained=False,search_size=256,template_size=128,
              template_number=1,neck_type="FB", **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        search_size=search_size, template_size=template_size, template_number=template_number, neck_type=neck_type,
        **kwargs)
    if pretrained:
        weight = "https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth"
        load_pretrained(model,weight)

    return model

@register_model
def pvit_large(pretrained=True,search_size=256,template_size=128,
              template_number=1,neck_type="FB",**kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        search_size = search_size,template_size=template_size,template_number=template_number,neck_type=neck_type,
        **kwargs)
    if pretrained:
        weight = "https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth"
        load_pretrained(model,weight)

    return model
