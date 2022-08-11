"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from lib.utils.misc import is_main_process
# from .position_encoding import build_position_encoding
from lib.models.stark import resnet as resnet_module
from lib.models.stark.repvgg import get_RepVGG_func_by_name
from lib.models.vt import levit as levit_module
# from lib.models.vt import levit_ori as levit_module
import os


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, open_layers: list, num_channels: int, return_interm_layers: bool,
                 net_type="resnet"):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in backbone.named_parameters():
            # if not train_backbone or 'layer2' not in name and 'layer3' not in name:

            if not train_backbone:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
                    #print('freeze %s'%name)
                #print('the unfrozen layers in backbone:')
                #print([n for n, p in backbone.named_parameters() if p.requires_grad])

        # if return_interm_layers:
        #     if net_type == "resnet":
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2"}  # stride = 4, 8, 16
        #     elif net_type == "repvgg":
        #         return_layers = {"stage1": "0", "stage2": "1", "stage3": "2"}
        #     else:
        #         raise ValueError()
        # else:
        #     if net_type == "resnet":
        #         return_layers = {'layer3': "0"}  # stride = 16
        #     elif net_type == "repvgg":
        #         return_layers = {'stage3': "0"}  # stride = 16
        #     else:
        #         raise ValueError()
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)  # method in torchvision
        self.body = backbone
        self.num_channels = num_channels

    # def forward(self, tensor_list: NestedTensor):
    #     xs = self.body(tensor_list.tensors)
    #     out: Dict[str, NestedTensor] = {}
    #     for name, x in xs.items():
    #         m = tensor_list.mask
    #         assert m is not None
    #         mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
    #         out[name] = NestedTensor(x, mask)
    #     return out

    def forward(self, images_list):
        xs = self.body(images_list)
        # out = {}
        # for name, x in xs.items():
        #     out[name] = x
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 freeze_bn: bool,
                 neck_type: str,
                 open_layers: list,
                 ckpt_path=None):
        if "resnet" in name:
            norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
            # here is different from the original DETR because we use feature from block3
            backbone = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, dilation, False],
                pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
            num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
            net_type = "resnet"
        elif "RepVGG" in name:
            print("#" * 10 + "  Freeze_BN and Dilation are not supported in current code  " + "#" * 10)
            # here is different from the original DETR because we use feature from block3
            repvgg_func = get_RepVGG_func_by_name(name)
            backbone = repvgg_func(deploy=False, last_layer="stage3")
            num_channels = 192  # 256x0.75=192
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                missing_keys, unexpected_keys = backbone.load_state_dict(ckpt, strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
            except:
                print("Warning: Pretrained RepVGG weights are not loaded")
            net_type = "repvgg"
        elif "vit" in name.lower():
            # norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
            # todo: frozenlayernorm
            if "LeViT" in name:
                # fuse == False when training
                backbone = getattr(levit_module, name)(
                    num_classes=0,
                    distillation=False,
                    pretrained=is_main_process(),
                    fuse = False,
                    search_size=search_size,
                    template_size=template_size,
                    template_number=template_number,
                    neck_type=neck_type
                )
                if "LeViT_128S" in name:
                    num_channels = 384
                elif "LeViT_128" in name:
                    num_channels = 384
                elif "LeViT_192" in name:
                    num_channels = 384
                elif "LeViT_256" in name:
                    num_channels = 512
                elif "LeViT_384" in name:
                    num_channels = 768
                else:
                    num_channels = 768
                net_type = "levit"
        else:
            raise ValueError()
        super().__init__(backbone, train_backbone, open_layers, num_channels, return_interm_layers, net_type=net_type)



def build_backbone(cfg):
    train_backbone = (cfg.TRAIN.BACKBONE_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_BACKBONE == False)
    return_interm_layers = cfg.MODEL.PREDICT_MASK
    if cfg.MODEL.BACKBONE.TYPE == "RepVGG-A0":
        try:
            ckpt_path = os.path.join(cfg.ckpt_dir, "RepVGG-A0-train.pth")
        except:
            ckpt_path = None
    else:
        ckpt_path = None
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                        cfg.MODEL.BACKBONE.DILATION, cfg.MODEL.BACKBONE.PRETRAIN_TYPE,
                        cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                        cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                        cfg.TRAIN.FREEZE_BACKBONE_BN, cfg.MODEL.NECK.TYPE, cfg.TRAIN.BACKBONE_OPEN,
                        ckpt_path)
    model = backbone
    model.num_channels = backbone.num_channels
    return model
