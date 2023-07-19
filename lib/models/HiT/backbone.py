"""
Backbone modules.
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.HiT import levit as levit_module
from lib.models.HiT import pvt as pvt_module
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

            if not train_backbone:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, images_list):
        xs = self.body(images_list)
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
        if "vit" in name.lower():
            # todo: frozenlayernorm
            if "pvit" in name:
                backbone = getattr(pvt_module,name)(
                    pretrained=is_main_process(),
                    search_size=search_size,template_size=template_size,
                    template_number=template_number,neck_type=neck_type
                )
                num_channels = 512
                net_type = "pvit"

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
