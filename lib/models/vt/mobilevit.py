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


def Conv_BN_ReLU(inp, oup, kernel, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        q, k, v = map(lambda t: rearrange(t, 'b  n (h d) -> b  h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b p h n d -> b p n (h d)')
        out = rearrange(out, 'b  h n d -> b  n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MV2Block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv_BN_ReLU(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        # self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        self.transformer = Transformer(dim, depth, 4, 32, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = Conv_BN_ReLU(2 * channel, channel, kernel_size)

    def forward(self, x,res_out):#  x==>image_list[search,template0,...]

        # y = x.clone()
        x_out = []
        x_fout= []
        for i in range(len(x)):
            xm = x[i]
        # Local representations
            xm = self.conv1(xm)
            xm = self.conv2(xm)
            b,c,_,_ = xm.shape
            x_out.append(xm)
            if i == 0:
                xz = rearrange(xm, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw).view(b*self.ph*self.pw,-1,c)#bs,h*w,c
                # xz = rearrange(xm, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
            else:
                xz = torch.cat((xz,rearrange(xm, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw).view(b*self.ph*self.pw,-1,c)),dim=1)
                # xz = torch.cat((xz,rearrange(xm, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)), dim=2)
        xz = self.transformer(xz)
        b,c,h_x,w_x = x_out[0].shape
        _,_,h_z,w_z = x_out[-1].shape
        for i in range(len(x)):
            if i == 0:
                xm = xz[:,:int(h_x*w_x/4),:].view(b,self.ph*self.pw,-1,c)
                # xm = xz[:, :,:h_x * w_x, :]
                xm = rearrange(xm, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h_x // self.ph, w=w_x // self.pw, ph=self.ph,
                              pw=self.pw)
            else:
                xm = xz[:,(int(h_x*w_x/4+(i-1)*h_z*w_z/4)):int(h_x*w_x/4+i*h_z*w_z/4),:].view(b,self.ph*self.pw,-1,c)
                xm = rearrange(xm, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h_z // self.ph, w=w_z // self.pw, ph=self.ph,
                              pw=self.pw)
            xm = self.conv3(xm)
            xm = torch.cat((xm,x[i]),1)
            xm = self.conv4(xm)
            x_fout.append(xm)
        res_out.append(x_fout[0].flatten(2).permute(0,2,1))#[x]

        return x_fout,res_out


class MobileViT(nn.Module):
    def __init__(self,  dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2),
                 search_size=256, template_size=128,
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
        ph, pw = patch_size
        assert self.search_size % ph == 0 and self.template_size % pw == 0

        L = [2, 4, 3]

        self.conv1 = Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])
        #
        # self.pool = nn.AvgPool2d(ih // 32, 1)
        # self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, image_list):#  image_list[search,template0,...]
        out = []
        out1 = []
        out2 = []
        res_out = []
        for i in range(self.template_number+1):
            x = image_list[i]
            x = self.conv1(x)
            x = self.mv2[0](x)

            x = self.mv2[1](x)
            x = self.mv2[2](x)
            x = self.mv2[3](x)  # Repeat

            x = self.mv2[4](x)
            out.append(x)

        x , res_out = self.mvit[0](out,res_out)
        for i in range(len(image_list)):
            xm = x[i]
            xm = self.mv2[5](xm)
            out1.append(xm)
        x ,res_out = self.mvit[1](out1,res_out)
        for i in range(len(image_list)):
            xm = x[i]
            xm = self.mv2[6](xm)
            out2.append(xm)
        #x[i] = b,c,h,w;res[i] = b,c,hw
        x ,res_out= self.mvit[2](out2,res_out)#x==>list[x,z],res_out==>list[stage_i_out]
        xf = self.conv2(x[0])
        res_out[-1] = xf.flatten(2).permute(0,2,1)
        cls = xf.flatten(2).permute(0,2,1).mean(1).unsqueeze(1)
        res_out.append(cls)#(stage1,stage2,stage3,cls)
        return res_out


def load_pretrained(model,model_name):

    url = "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt"
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu', check_hash=False,
    )
    state_dict = checkpoint['model']
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


def mobilevit_s(pretrained=False,num_classes=1000,
                search_size=256,template_size=128,
                template_number=1,neck_type="FPN"):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    embed_dim_list = [96,128,640]
    model = MobileViT( dims, channels, num_classes=num_classes,
                      search_size=search_size,template_size=template_size,
                      template_number=template_number,neck_type=neck_type,embed_dim_list=embed_dim_list)
    if pretrained:
        load_pretrained(model)

    return model