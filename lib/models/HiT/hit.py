"""
Basic STARK Model (Spatial-only).
"""
import torch
import math
from torch import nn

from .backbone import build_backbone
from .head import build_box_head
from .neck import build_neck
from lib.utils.box_ops import box_xyxy_to_cxcywh


class HiT(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, box_head, hidden_dim, num_queries,
                 bottleneck=None, aux_loss=False, head_type="CORNER", neck_type='LINEAR'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.num_patch_x = self.backbone.body.num_patches_search
        self.num_patch_z = self.backbone.body.num_patches_template
        self.neck_type = neck_type
        if neck_type in ['UPSAMPLE', 'FB','MAXF','MAXMINF','MAXMIDF','MINMIDF','MIDF','MINF']:
            self.num_patch_x = self.backbone.body.num_patches_search * ((bottleneck.stride_total) ** 2)
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.box_head = box_head
        self.num_queries = num_queries
        if bottleneck == None:
            self.bottleneck = nn.Linear(backbone.num_channels, hidden_dim) # the bottleneck layer
        else:
            self.bottleneck = bottleneck
        self.aux_loss = aux_loss
        self.head_type = head_type
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, images_list=None, xz=None, mode="backbone", run_box_head=True, run_cls_head=False,first_score=None,threshold=0.9,frame=True,score_t=0.6,xz_mem=0):
        if mode == "backbone":
            return self.forward_backbone(images_list,first_score,threshold,frame,score_t=score_t)
        elif mode == "head":
            return self.forward_head(xz, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, images_list,first_score,threshold,frame=True,score_t=0.6):
        # Forward the backbone
        xz = self.backbone(images_list,first_score,threshold,frame,score_t=score_t)   # features & masks, position embedding for the search
        return xz

    def forward_head(self, xz, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        if self.neck_type == 'FB' or self.neck_type == "MAXF" or self.neck_type == 'MAXMINF' or self.neck_type == "MAXMIDF" or self.neck_type == 'MINMIDF' or self.neck_type == "MIDF" or self.neck_type == "MINF":
            xz_mem = self.bottleneck(xz)
        else:
            xz_mem = xz[-1].permute(1, 0, 2)
            xz_mem = self.bottleneck(xz_mem)
        output_embed = xz_mem[0:1,:,:].unsqueeze(-2)
        x_mem = xz_mem[1:1+self.num_patch_x]
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, x_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if "CORNER" in self.head_type and self.head_type != "CORNER_WOATT":
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "CORNER_WOATT":
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1).permute(0,2,1)#(B,C,HW)
            bs, C, HW = enc_opt.size()
            opt_feat = enc_opt.view(-1,C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord


    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

class DyHiT(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self,cfg, model1,box_head_small,bottleneck_small):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cfg = cfg
        self.model1 = model1
        self.box_head_small = box_head_small
        self.bottleneck_small = bottleneck_small
        self.num_patch_x = 256
        self.feat_sz_s = int(self.box_head_small.feat_sz)
        self.feat_len_s = int(self.feat_sz_s ** 2)
        # for DyHiT stage2
        if cfg.TRAIN.STAGE == 2:
            for n,p in self.named_parameters():
                if 'router' not in n:
                    p.requires_grad_(False)

    def forward(self, images_list=None, run_box_head=True, run_cls_head=False,first_score=None,threshold=0.9,frame=True,score_t=0.6):
        feature_xz1 = self.model1(images_list=images_list, mode='backbone',first_score=first_score,threshold=threshold,frame=frame,score_t=score_t)#stage2 out [smax,Smid,Smin,router_cls]
        if self.model1.training:
            if len(feature_xz1) == 1:# use small_tracker
                xz_mem = feature_xz1[-1].permute(1, 0, 2)
                xz_mem = self.bottleneck_small(xz_mem)
                output_embed = xz_mem[0:1, :, :].unsqueeze(-2)
                x_mem = xz_mem[1:1 + self.num_patch_x]
                out, outputs_coord = self.forward_box_head(output_embed, x_mem)
                return out
            else:
                xz_mem = feature_xz1[0].permute(1, 0, 2)
                xz_mem = self.bottleneck_small(xz_mem)
                output_embed = xz_mem[0:1, :, :].unsqueeze(-2)
                x_mem = xz_mem[1:1 + self.num_patch_x]
                out, outputs_coord = self.forward_box_head(output_embed, x_mem)
                return out,feature_xz1[-1]
        #验证
        else:
            if len(feature_xz1) == 2:# use small_tracker
                xz_mem = feature_xz1[0].permute(1, 0, 2)
                xz_mem = self.bottleneck_small(xz_mem)
                output_embed = xz_mem[0:1, :, :].unsqueeze(-2)
                x_mem = xz_mem[1:1 + self.num_patch_x]
                out, outputs_coord = self.forward_box_head(output_embed, x_mem)
                return out, feature_xz1[-1]
            else:
            #第二阶段
                feature_stage3 = [feature_xz1[0],feature_xz1[1],feature_xz1[2]]
                #
                out, _, _ = self.model1(xz=feature_stage3, mode="head", run_box_head=run_box_head,
                                                  run_cls_head=run_cls_head)
                return out,feature_xz1[-1]

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        # adjust shape
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head_small(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new


def build_hit(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)
    bottleneck = build_neck(cfg, backbone.num_channels, backbone.body.num_patches_search, backbone.body.embed_dim_list)
    model = HiT(
        backbone,
        box_head,
        bottleneck = bottleneck,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        neck_type=cfg.MODEL.NECK.TYPE
    )

    return model

from collections import OrderedDict
def load_pretrained(model, cfg):
    weights = cfg.TRAIN.WEIGHT
    checkpoint = torch.load(
        weights, map_location='cpu')
    state_dict = checkpoint['net']
    state_dict_load = OrderedDict()
    if cfg.TRAIN.STAGE == 1:
        for key in state_dict.keys():
            if key in model.model1.state_dict().keys():
                state_dict_load[key] = state_dict[key]
            else:
                print("loss key:",key)
        model.model1.load_state_dict(state_dict_load,strict=True)
    if cfg.TRAIN.STAGE == 2:
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                state_dict_load[key] = state_dict[key]
            elif 'router' in key:
                state_dict_load[key] = model.state_dict()[key]
            else:
                print("loss key:", key)
        model.load_state_dict(state_dict_load, strict=True)
    return model


def build_dyhit(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)
    if cfg.MODEL.NECK.TYPE :
        bottleneck = build_neck(cfg, backbone.num_channels, backbone.body.num_patches_search, backbone.body.embed_dim_list)
    else:
        bottleneck = None
    model_big = HiT(
        backbone,
        box_head,
        bottleneck = bottleneck,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        neck_type=cfg.MODEL.NECK.TYPE
    )
    if cfg.TRAIN.STAGE == 1 :
        for p in model_big.parameters():
            p.requires_grad_(False)
    bottleneck_small = nn.Linear(384, cfg.MODEL.HIDDEN_DIM)
    box_head_small = build_box_head(cfg)
    model = DyHiT(cfg,model1=model_big,box_head_small=box_head_small, bottleneck_small=bottleneck_small)
    model = load_pretrained(model,cfg)
    return model
