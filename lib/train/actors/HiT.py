from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import numpy as np


class HiTActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        # process the templates
        images_list = []
        # process the search regions (t-th frame)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 384, 384)
        images_list.append(search_img)
        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 192, 192)
            images_list.append(template_img)
        feature_xz = self.net(images_list=images_list, mode='backbone')

        out_dict, _, _ = self.net(xz=feature_xz, mode="head", run_box_head=run_box_head, run_cls_head=run_cls_head)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss


class DyHiTActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        # process the templates
        images_list = []
        # process the search regions (t-th frame)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 384, 384)
        images_list.append(search_img)
        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 192, 192)
            images_list.append(template_img)
        out_dict = self.net(images_list=images_list, run_box_head=run_box_head, run_cls_head=run_cls_head)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

class DyHiTActor_stage2(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

        self.iter_count = 0
        self.diff_list = []
        self.tracked_diff = 0
        self.label = 0

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_small,score = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        # loss, status = self.compute_losses(out_dict, gt_bboxes[0])
        loss, status = self.compute_losses(out_small,score, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        # process the templates
        images_list = []
        # process the search regions (t-th frame)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 384, 384)
        images_list.append(search_img)
        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 192, 192)
            images_list.append(template_img)
        out_dict = self.net(images_list=images_list, run_box_head=run_box_head, run_cls_head=run_cls_head)
        return out_dict

    def compute_losses(self, pred_dict, score, gt_bbox, return_status=True):
        bs,num_iou,_ = score.size()
        score_map = torch.zeros((bs,num_iou)).cuda()
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou_1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.zeros(bs).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1_1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        #get positive_box indices
        indices = self.matcher(bs,num_iou,gt_boxes_vec)
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(pred_dict.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()
        # iou_score = score[idx]
        for i,indice in enumerate(indices):
            inic,id = indice
            i_ = [i]*len(id)
            iou_ = iou[i_]
            score_map[i][inic] = iou_
        # target_iou = torch.cat([t[i] for t, (_, i) in zip(iou, indices)], dim=0)
        # weighted sum
        criterion = torch.nn.MSELoss(reduction='sum')# mse for iou
        iou_loss = criterion(score_map, score.squeeze(),)
        iou_loss = iou_loss/num_boxes_pos
        loss = 5 * iou_loss + l1_loss + giou_loss
        # print('yesyes!!!!!!!!!!!1')
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/iou": iou_loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def matcher(self,bs,num_queries,targets):
        indices = []
        for i in range(bs):
            xmin, ymin, xmax, ymax = targets[i]
            xmin = xmin.item()
            ymin = ymin.item()
            xmax = xmax.item()
            ymax = ymax.item()
            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin*len_feature))
            Ymin = int(np.ceil(ymin*len_feature))
            Xmax = int(np.ceil(xmax*len_feature))
            Ymax = int(np.ceil(ymax*len_feature))
            if Xmin == Xmax:
                Xmax = Xmax+1
            if Ymin == Ymax:
                Ymax = Ymax+1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax,Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c,d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]