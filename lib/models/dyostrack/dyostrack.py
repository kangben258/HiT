"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.dyostrack.layers.head import build_box_head
from lib.models.dyostrack.vit import vit_base_patch16_224
from lib.models.dyostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.HiT.backbone import Backbone
from lib.models.HiT.head import Corner_Predictor

def build_box_head_small(cfg):
    stride = cfg.MODEL2.BACKBONE.STRIDE
    # stride = stride // (2 * 2)
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    channel = getattr(cfg.MODEL2, "HEAD_DIM", 256)
    print("head channel: %d" % channel)
    if cfg.MODEL2.HEAD_TYPE == "CORNER":
        corner_head = Corner_Predictor(inplanes=cfg.MODEL2.HIDDEN_DIM, channel=channel,
                                       feat_sz=feat_sz, stride=stride)
    else:
        raise ValueError()
    return corner_head
class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

class Model(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, images_list=None, xz=None, mode="backbone", run_box_head=True, run_cls_head=False,first_score=None,threshold=0.9):
        if mode == "backbone":
            return self.forward_backbone(images_list,first_score,threshold)
    def forward_backbone(self, images_list,first_score,threshold):
        # Forward the backbone
        xz = self.backbone(images_list,first_score,threshold)  # features & masks, position embedding for the search
        return xz

class HiT(nn.Module):
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
    def forward(self, images_list=None, run_box_head=True, run_cls_head=False,first_score=None,threshold=0.9):
        feature_xz1 = self.model1(images_list=images_list, mode='backbone',first_score=first_score,threshold=threshold)#stage2 out [smax,Smid,Smin,router_cls]
        return feature_xz1

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


class DYOSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, model_os, model_hit, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.model_os = model_os
        self.model_hit = model_hit

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(model_os.box_head.feat_sz)
            self.feat_len_s = int(model_os.box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                threshold=0.9
                ):
        img_list = [search,template]
        model_hit_out = self.model_hit(img_list)#[out,score]
        score = model_hit_out[-1]
        select_element = score[score > 0.6]
        score = torch.mean(select_element).item()

        #认为当前帧简单可以使用hit
        if score > threshold:
            xz_mem = model_hit_out[0].permute(1, 0, 2)
            xz_mem = self.model_hit.bottleneck_small(xz_mem)
            output_embed = xz_mem[0:1, :, :].unsqueeze(-2)
            x_mem = xz_mem[1:1 + self.model_hit.num_patch_x]
            out, outputs_coord = self.model_hit.forward_box_head(output_embed, x_mem)
            return out, score
        #使用Ostrack
        else:
            x, aux_dict = self.model_os.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            out = self.model_os.forward_head(feat_last, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out,score

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_dyostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL1.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL1.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL1.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL1.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL1.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL1.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL1.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL1.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL1.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL1.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    model_os = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL1.HEAD.TYPE,
    )
    #load checkpoint
    checkpoint = torch.load(cfg.MODEL1.WEIGHT, map_location="cpu")
    model_os.load_state_dict(checkpoint["net"], strict=True)

    # hit
    backbone_hit = Backbone(cfg.MODEL2.BACKBONE.TYPE, True, False,
                        cfg.MODEL2.BACKBONE.DILATION, cfg.MODEL2.BACKBONE.PRETRAIN_TYPE,
                        cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                        cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                        True, cfg.MODEL2.NECK.TYPE, [],
                        None)
    model_backbone = Model(backbone_hit)
    bottleneck_hit = nn.Linear(384, cfg.MODEL2.HIDDEN_DIM)
    box_head_hit = build_box_head_small(cfg)
    model_hit = HiT(cfg,model_backbone,box_head_hit,bottleneck_hit)
    checkpoint = torch.load(cfg.MODEL2.WEIGHT, map_location="cpu")
    missing_keys, unexpected_keys = model_hit.load_state_dict(checkpoint["net"], strict=False)
    # print("model_hit missing_keys:",missing_keys)
    # print("model_hit unexpected_keys:", unexpected_keys)

    model = DYOSTrack(
        model_os,
        model_hit
    )


    return model

