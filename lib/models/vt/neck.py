import torch.nn as nn
import torch
import torch.nn.functional as F

class NECK_UPSAMPLE(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, BN=False):
        super().__init__()
        self.num_x = num_x
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        st = [stride] * (num_layers)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))
                                        for n, k,s in zip([input_dim] + h, h + [hidden_dim], st))
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        for n, k,s in zip([input_dim] + h, h + [hidden_dim], st))
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz):
        global_vector = xz[0:1,:,:]
        x = xz[1:self.num_x+1,:,:]
        N,B,C = x.shape
        Len = int(N ** 0.5)
        x = x.permute(1, 2, 0).view(B, C, Len, Len)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = x.view(B,C,N*(((2**self.num_layers))**2)).permute(2, 0, 1)
        xz = torch.cat((global_vector, x), dim=0)
        xz = self.proj(xz)
        return xz

class NECK_FB(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        st = [stride] * (num_layers)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))#upsample
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)#global vector
        fb_features = []
        for i in range(len(xz_list)):
            if i == len(xz_list)-1:
                x = xz_list[i][:,1:self.num_x+1,:]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
            else:
                x = xz_list[i][:, 0:self.num_x*(4**(len(xz_list)-1-i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = x + fb_features[-2-i]
            if i < self.num_layers - 1:
                x = F.relu(x)
        B, C, Len, _ = x.shape
        x = x.view(B,C,Len*Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector= self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

class NECK_FB_PVIT(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        st = [stride] * (num_layers)
        n = backbone_embed_dim[-1]
        k = backbone_embed_dim[-2]
        s = st[0]
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s, s), (s, s)), nn.BatchNorm2d(k))
                                        #
                                        )
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s, s), (s, s)))
                                        )
        self.proj1 = nn.Linear(self.backbone_embed_dim[-2], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1].permute(1,0,2)
        fb_features = []
        for i in range(len(xz_list)):
            if i == 2 or i == 3:
                x = xz_list[i]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = x + fb_features[i]
        B, C, Len, _ = x.shape
        x = x.view(B, C, Len * Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector= self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

class NECK_MAXF(nn.Module):

    def __init__(self, num_x, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)
        x = xz_list[0][:, 0:self.num_x * (4 ** (len(xz_list) - 1 )), :].permute(1,0,2)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

class NECK_MAXMINF(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        st = [stride] * (num_layers)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)
        fb_features = []
        for i in range(len(xz_list)):
            if i == len(xz_list)-1:
                x = xz_list[i][:,1:self.num_x+1,:]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
            elif i == 0:
                x = xz_list[i][:, 0:self.num_x*(4**(len(xz_list)-1-i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.num_layers -1:
                x = x + fb_features[-1 - i]
            if i < self.num_layers - 1:
                x = F.relu(x)
        B, C, Len, _ = x.shape
        x = x.view(B,C,Len*Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

class NECK_MAXMIDF(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        st = [stride] * (num_layers)
        n = backbone_embed_dim[1]
        k = backbone_embed_dim[0]
        s = st[0]
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))
                                        )
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        )
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)
        fb_features = []
        for i in range(len(xz_list)):
            if i ==0 or i == 1:
                x = xz_list[i][:, 0:self.num_x*(4**(len(xz_list)-1-i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = x + fb_features[0]
        B, C, Len, _ = x.shape
        x = x.view(B,C,Len*Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

class NECK_MINMIDF(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        st = [stride] * (num_layers)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        for n, k,s in zip(backbone_embed_dim[::-1][0:-1], backbone_embed_dim[::-1][1:], st))
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)
        fb_features = []
        for i in range(len(xz_list)):
            if i == len(xz_list)-1:
                x = xz_list[i][:,1:self.num_x+1,:]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
            elif i == 1 :
                x = xz_list[i][:, 0:self.num_x*(4**(len(xz_list)-1-i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = x + fb_features[0]
            if i < self.num_layers - 1:
                x = F.relu(x)
        B, C, Len, _ = x.shape
        x = x.view(B,C,Len*Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz


class NECK_MIDF(nn.Module):

    def __init__(self, num_x, input_dim, hidden_dim, output_dim, stride, num_layers, backbone_embed_dim, BN=False):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.num_x = num_x
        self.num_layers = num_layers
        # h = [hidden_dim] * (num_layers - 1)
        st = [stride] * (num_layers)
        n = backbone_embed_dim[1]
        k = backbone_embed_dim[0]
        s = st[0]
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)), nn.BatchNorm2d(k))
                                        )
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(n, k, (s,s), (s,s)))
                                        )
        self.proj1 = nn.Linear(self.backbone_embed_dim[0], output_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.stride_total = stride ** num_layers

    def forward(self, xz_list):
        global_vector = xz_list[-1][:,0:1,:].permute(1,0,2)
        fb_features = []
        for i in range(len(xz_list)):
            if i ==0 or i == 1:
                x = xz_list[i][:, 0:self.num_x*(4**(len(xz_list)-1-i)), :]
                B, N, C = x.shape
                Len = int(N ** 0.5)
                x = x.permute(0, 2, 1).view(B, C, Len, Len)
                fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
        B, C, Len, _ = x.shape
        x = x.view(B,C,Len*Len).permute(2, 0, 1)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=0)
        return xz

def build_neck(cfg, backbone_channels, num_x, backbone_embed_dim_list):
    if "pvit" in cfg.MODEL.BACKBONE.TYPE:
        neck = NECK_FB_PVIT(
            num_x, backbone_channels, backbone_channels,
            cfg.MODEL.HIDDEN_DIM,
            cfg.MODEL.NECK.STRIDE,
            cfg.MODEL.NECK.NUM_LAYERS,
            backbone_embed_dim_list,
            BN=True
        )
        return neck

    else:
        if cfg.MODEL.NECK.TYPE == "LINEAR":
            neck = nn.Linear(backbone_channels, cfg.MODEL.HIDDEN_DIM)
            return neck
        elif cfg.MODEL.NECK.TYPE == "UPSAMPLE":
            neck = NECK_UPSAMPLE(num_x, backbone_channels, backbone_channels, cfg.MODEL.HIDDEN_DIM,
                                 cfg.MODEL.NECK.STRIDE, cfg.MODEL.NECK.NUM_LAYERS, BN=True)
            return neck
        elif cfg.MODEL.NECK.TYPE == "FB":
            neck = NECK_FB(num_x, backbone_channels, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list,
                            BN=True)
            return neck
        elif cfg.MODEL.NECK.TYPE == "MAXF":
            neck = NECK_MAXF(num_x, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list)
            return neck
        elif cfg.MODEL.NECK.TYPE == "MAXMINF":
            neck = NECK_MAXMINF(num_x, backbone_channels, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list,
                            BN=True)
            return neck
        elif cfg.MODEL.NECK.TYPE == "MAXMIDF":
            neck = NECK_MAXMIDF(num_x, backbone_channels, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list,
                            BN=True)
            return neck
        elif cfg.MODEL.NECK.TYPE == "MINMIDF":
            neck = NECK_MINMIDF(num_x, backbone_channels, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list,
                            BN=True)
            return neck
        elif cfg.MODEL.NECK.TYPE == "MIDF":
            neck = NECK_MIDF(num_x, backbone_channels, backbone_channels,
                            cfg.MODEL.HIDDEN_DIM,
                            cfg.MODEL.NECK.STRIDE,
                            cfg.MODEL.NECK.NUM_LAYERS,
                            backbone_embed_dim_list,
                            BN=True)
            return neck
        else:
            raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
