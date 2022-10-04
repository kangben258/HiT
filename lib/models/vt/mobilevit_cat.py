# origin code: https://github.com/murufeng/awesome_lightweight_networks/blob/3917c7f919bdd5c445b07e6df617f96f1392321f/light_cnns/Transformer/mobile_vit.py#L85

import torch
import torch.nn as nn

from einops import rearrange
from collections import OrderedDict

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv_BN_ReLU(nn.Module):
    def __init__(self,inp,oup,kernel,stride=1):
        super().__init__()
        block = nn.Sequential()
        conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False)
        block.add_module(name="conv", module=conv_layer)
        norm_layer = nn.BatchNorm2d(oup)
        block.add_module(name="norm", module=norm_layer)
        block.add_module(name="act", module=nn.SiLU())
        self.block = block
    def forward(self,x):
        return self.block(x)

# def Conv_BN_ReLU(inp, oup, kernel, stride=1):
#     block = nn.Sequential()
#     conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False)
#     block.add_module(name="conv",module=conv_layer)
#     norm_layer = nn.BatchNorm2d(oup)
#     block.add_module(name="norm",module=norm_layer)
#     block.add_module(name="act",module=nn.ReLU6(inplace=True))
#
#     return block
class Conv_BN_SILU(nn.Module):
    def __init__(self,inp,oup, kernel, stride=1,use_silu=False):
        super().__init__()
        block = nn.Sequential()
        if kernel == 3:
            conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, groups=oup, padding=1, bias=False)
        else:
            conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=0, bias=False)
        block.add_module(name="conv", module=conv_layer)
        norm_layer = nn.BatchNorm2d(oup)
        block.add_module(name="norm", module=norm_layer)
        if use_silu:
            block.add_module(name="act", module=nn.SiLU())
        self.block =block
    def forward(self,x):
        return self.block(x)


# def Conv_BN_SILU(inp, oup, kernel, stride=1,use_silu=False):
#     block = nn.Sequential()
#     if kernel == 3:
#         conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, groups=oup,padding=1, bias=False)
#     else:
#         conv_layer = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=0, bias=False)
#     block.add_module(name="conv",module=conv_layer)
#     norm_layer = nn.BatchNorm2d(oup)
#     block.add_module(name="norm",module=norm_layer)
#     if use_silu:
#         block.add_module(name="act",module=nn.SiLU())
#
#     return block

class conv_1x1(nn.Module):
    def __init__(self,inp,oup):
        super().__init__()
        block = nn.Sequential()
        conv_layer = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        block.add_module(name="conv", module=conv_layer)
        self.block = block
    def forward(self,x):
        return self.block(x)
# def conv_1x1(inp,oup):
#     block = nn.Sequential()
#     conv_layer = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
#     block.add_module(name="conv",module=conv_layer)
#     return block

class conv_1x1_bn(nn.Module):
    def __init__(self,inp,oup,use_act=True):
        super().__init__()
        block = nn.Sequential()
        conv_layer = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        block.add_module(name="conv",module=conv_layer)
        norm_layer = nn.BatchNorm2d(oup)
        block.add_module(name="norm",module=norm_layer)
        if use_act:
            block.add_module(name="act",module=nn.SiLU())
        self.block = block
    def forward(self,x):
        return self.block(x)
# def conv_1x1_bn(inp, oup):
#     block = nn.Sequential()
#     conv_layer = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
#     block.add_module(name="conv",module=conv_layer)
#     norm_layer = nn.BatchNorm2d(oup)
#     block.add_module(name="norm",module=norm_layer)
#     block.add_module(name="act",module=nn.ReLU6(inplace=True))
#     return block

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim,dim,bias=True)
        self.head_dim = dim//heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = heads
        self.embed_dim = dim



        # inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)
        #
        # self.heads = heads
        # self.scale = dim_head ** -0.5
        #
        # self.attend = nn.Softmax(dim=-1)
        # self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        # self.out_proj = nn.Linear(inner_dim, dim)
        #
        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        q, k, v = map(lambda t: rearrange(t, 'b  n (h d) -> b  h n d', h=self.num_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn = self.softmax(dots)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = rearrange(out, 'b  h n d -> b  n (h d)')
        out = self.out_proj(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        attn_unit = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(dim),
            attn_unit,
            nn.Dropout(0.1)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.1)

        )




        # self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ]))
    def forward(self, x):
        #mha
        res = x
        x = self.pre_norm_mha[0](x)
        x = self.pre_norm_mha[1](x)
        x = self.pre_norm_mha[2](x)
        x = x+res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MV2Block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        block = nn.Sequential()

        # if expand_ratio == 1:
        #     # self.conv = nn.Sequential(
        #     #     # dw
        #     #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        #     #     nn.BatchNorm2d(hidden_dim),
        #     #     #nn.ReLU6(inplace=True),
        #     #     nn.SiLU(),
        #     #     # pw-linear
        #     #     nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #     #     nn.BatchNorm2d(oup),
        #     # )
        # else:
        #     self.conv = nn.Sequential(
        #         # pw
        #         nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #         #nn.ReLU6(inplace=True),
        #         nn.SiLU(),
        #         # dw
        #         nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #         #nn.ReLU6(inplace=True),
        #         nn.SiLU(),
        #         # pw-linear
        #         nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(oup),
        #     )
        if expand_ratio !=1:
            block.add_module(
                name="exp_1x1",
                module=Conv_BN_SILU(inp, hidden_dim, kernel=1, stride=1,use_silu=True)
            )
        block.add_module(
            name="conv_3x3",
            module=Conv_BN_SILU(hidden_dim, hidden_dim, kernel=3, stride=stride,use_silu=True)
        )
        block.add_module(
            name="red_1x1",
            module=Conv_BN_SILU(hidden_dim,oup,kernel=1,stride=1,use_silu=False)
        )
        self.block = block

    def forward(self, x,res_out):
        if self.identity:
            return x + self.block(x),res_out
        else:
            return self.block(x),res_out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        conv1 = Conv_BN_ReLU(channel, channel, kernel_size)
        conv2 = conv_1x1(channel, dim)
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3",module=conv1)
        self.local_rep.add_module(name="conv_1x1",module=conv2)
        global_rep = [
            Transformer(dim, 4, 32, mlp_dim, dropout)
            for _ in range(depth)
        ]
        global_rep.append(
            nn.LayerNorm(dim)
        )
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_bn(dim,channel)
        self.fusion = Conv_BN_ReLU(2 * channel, channel, kernel_size)


        # self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        # self.transformer = Transformer(dim, depth, 4, 32, mlp_dim, dropout)
        #
        # conv3 = conv_1x1_bn(dim, channel)
        # conv4 = Conv_BN_ReLU(2 * channel, channel, kernel_size)

    def forward(self, x,res_out):#  x==>image[2b,c,h,w]

        num = 2
        xm = self.local_rep(x)
        b,c,h,w = xm.shape


        xz = rearrange(xm, '(b n) d (h ph) (w pw) -> b (ph pw) (n h w) d', ph=self.ph, pw=self.pw,n=num).view(b//num*self.ph*self.pw,-1,c)#bs,h*w,c
                # xz = rearrange(xm, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        for transformer_layer in self.global_rep:
            xz = transformer_layer(xz)

        xz = rearrange(xz.view(b//num,self.ph*self.pw,-1,c), 'b (ph pw) (n h w) d -> (b n) d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                       pw=self.pw)
        xz = self.conv_proj(xz)
        xz = torch.cat((xz,x),dim=1)
        xz = self.fusion(xz)
        res_out.append(xz[:int(b//num),:,:,:].flatten(2).permute(0,2,1))

        return xz,res_out


class MobileViT(nn.Module):
    def __init__(self,  dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2),
                 search_size=256, template_size=256,
                 template_number=1, neck_type="FPN",embed_dim_list = [96,128,640]
                 ):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.template_number = template_number
        self.neck_type = neck_type
        self.num_patches_search = (self.search_size//32)**2
        self.num_patches_template = (self.template_size // 32) ** 2
        self.num_classes = num_classes
        self.embed_dim_list = embed_dim_list
        self.channels = channels
        self.expansion = expansion
        ph, pw = patch_size
        assert self.search_size % ph == 0 and self.template_size % pw == 0

        L = [2, 4, 3]

        # self.conv1 = Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)
        #
        # self.mv2 = nn.ModuleList([])
        # #layer1
        # self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        # #layer2
        # self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        # self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        # self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion)) # Repeat
        # #layer3
        # self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        # # self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        # #layer 4
        # self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        # # self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        # #layer 5
        # self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        # # self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))
        #
        # self.mvit = nn.ModuleList([])
        # self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        # self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        # self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))
        #
        # self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        #
        # self.pool = nn.AvgPool2d(ih // 32, 1)
        # self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        self.model_conf_dict = dict()
        self.conv_1 = Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)
        self.model_conf_dict["conv1"] = {"in":3,"out":channels[0]}
        self.layer_1 = self.make_mv2layer(in_channels=[channels[0]],oup_channels=[channels[1]],stride=[1],expansion=expansion,num_blocks=1)
        self.model_conf_dict["layer1"] = {"in": channels[0], "out": channels[1]}
        self.layer_2 = self.make_mv2layer(in_channels=[channels[1],channels[2],channels[2]],
                                          oup_channels=[channels[2],channels[3],channels[3]],stride=[2,1,1],expansion=expansion,num_blocks=3)
        self.model_conf_dict["layer2"] = {"in": channels[1],"out": channels[3]}
        self.layer_3 = self.make_mobilevitlayer(dim=dims[0],length= L[0],channels=[channels[3],channels[4],channels[5]],kernel_size=kernel_size
                                                ,patch_size=patch_size,mlp_dim=int(dims[0] * 2))#channels(mv2_in_chann,mv2_out_chann,mbvit_chann)
        self.model_conf_dict['layer3'] = {'in': channels[3],'out': channels[5]}
        self.layer_4 = self.make_mobilevitlayer(dim=dims[1],length=L[1],channels=[channels[5],channels[6],channels[7]],
                                                kernel_size=kernel_size,patch_size=patch_size,mlp_dim=int(dims[1]*2))
        self.model_conf_dict['layer4'] = {'in': channels[5],'out':channels[7]}
        self.layer_5 = self.make_mobilevitlayer(dim=dims[2],length=L[2],channels=[channels[7],channels[8],channels[9]],
                                                kernel_size=kernel_size,patch_size=patch_size,mlp_dim=int(dims[2]*2))
        self.model_conf_dict['layer5'] = {'in': channels[7],'out':channels[9]}
        self.conv_1x1_exp = conv_1x1_bn(channels[-2],channels[-1],use_act=True)
        self.model_conf_dict['exp_before_cls'] = {
            'in':channels[-2],
            'out':channels[-1]
        }

    def make_mv2layer(self,in_channels,oup_channels,stride,expansion,num_blocks):
        block = []
        for i in range(num_blocks):
            layer = MV2Block(in_channels[i],oup_channels[i],stride=stride[i],expand_ratio=expansion)
            block.append(layer)
        return nn.Sequential(*block)

    def make_mobilevitlayer(self,dim,length,channels,kernel_size,patch_size,mlp_dim):
        block = []
        layer = MV2Block(channels[0],channels[1],stride=2,expand_ratio=self.expansion)
        block.append(layer)
        block.append(
            MobileViTBlock(dim, length, channels[2], kernel_size, patch_size, mlp_dim)
        )
        return nn.Sequential(*block)


    def forward(self, image_list):#  image_list[search,template0,...]
        res_out = []
        b,c,h,w = image_list[0].shape
        for i in range(self.template_number+1):
            if i==0:
                xz = image_list[i]
            else:
                xz = torch.cat((xz,image_list[i]),dim=0)#xz==>2b,c,h,w
        #image==>patches
        x = self.conv_1(xz)
        #layer1
        for layer in self.layer_1:
            x ,res_out = layer(x,res_out)
        #layer2
        for layer in self.layer_2:
            x ,res_out = layer(x,res_out)
        #layer3
        for layer in self.layer_3:
            x,res_out = layer(x,res_out)
        #layer4
        for layer in self.layer_4:
            x,res_out = layer(x,res_out)
        #layer5
        for layer in self.layer_5:
            x,res_out = layer(x,res_out)
        #last_conv
        xf = self.conv_1x1_exp(x[:b,:,:,:])

        res_out[-1] = xf.flatten(2).permute(0,2,1)
        cls = xf.flatten(2).permute(0, 2, 1).mean(1).unsqueeze(1)
        res_out.append(cls)
        return res_out

def load_pretrained(model):

    url = "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt"
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu', check_hash=False,
    )
    state_dict = checkpoint
    # s = model.state_dict()
    # model.load_state_dict(state_dict)
    state_dict_load = OrderedDict()
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            state_dict_load[key] = state_dict[key]#attention bias与图片大小有关不能直接load
    model.load_state_dict(state_dict_load)



def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s(pretrained=True,num_classes=1000,
                search_size=256,template_size=256,
                template_number=1,neck_type="FPN"):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    embed_dim_list = [96, 128, 640]
    model = MobileViT( dims, channels, num_classes=num_classes,
                      search_size=search_size,template_size=template_size,
                      template_number=template_number,neck_type=neck_type,embed_dim_list=embed_dim_list)
    if pretrained:
        load_pretrained(model)

    return model