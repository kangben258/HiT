import argparse
import torch
import _init_paths
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn as nn
import torch.nn.functional as F
# for onnx conversion and inference
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation.environment import env_settings
import lib.models.HiT.levit_utils as utils
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='HiT', help='script name')
    parser.add_argument('--config', type=str, default='HiT_Base', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz=256):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch


class VT(nn.Module):
    def __init__(self, backbone, bottleneck, box_head, head_type="CORNER", neck_type='LINEAR'):
        super(VT, self).__init__()
        self.backbone = backbone
        self.num_patch_x = self.backbone.body.num_patches_search
        self.num_patch_z = self.backbone.body.num_patches_template
        self.neck_type = neck_type
        if neck_type in ['UPSAMPLE', 'FB','MAXF','MAXMINF','MAXMIDF','MINMIDF','MIDF','MINF']:
            self.num_patch_x = self.backbone.body.num_patches_search * ((bottleneck.stride_total) ** 2)
        self.side_fx = int(self.num_patch_x ** 0.5)
        self.side_fz = int(self.num_patch_z ** 0.5)
        self.bottleneck = bottleneck
        self.box_head = box_head
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, search, template):
        # run the backbone
        img_list = [search,template]
        xz = self.backbone(img_list) # BxCxHxW
        if self.neck_type in ['FB','MAXF','MAXMINF','MAXMIDF','MINMIDF','MIDF','MINF']:
            xz_mem = self.bottleneck(xz)
        else:
            xz_mem = xz[-1].permute(1, 0, 2)
            xz_mem = self.bottleneck(xz_mem)
        output_embed = xz_mem[0:1,:,:].unsqueeze(-2)
        x_mem = xz_mem[1:1+self.num_patch_x]
        # adjust shape
        enc_opt = x_mem[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        return outputs_coord_new


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    # update cfg
    args = parse_args()
    yaml_fname = prj_path+'/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    save_name = prj_path+"/checkpoints/train/%s/%s/VT_ep%04d.onnx" % (args.script, args.config, cfg.TEST.EPOCH)
    # build the model
    model_module = importlib.import_module('lib.models.HiT')
    model_constructor = model_module.build_hit
    model = model_constructor(cfg)
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(save_dir,
                                       "./checkpoints/train/%s/%s/VT_ep%04d.pth.tar"
                                       % (args.script, args.config, cfg.TEST.EPOCH))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # merge conv+bn for levit
    if "LeViT" in cfg.MODEL.BACKBONE.TYPE:
        # merge conv+bn to one operator
        utils.replace_batchnorm(model.backbone.body)
    # transfer to test mode
    model.eval()
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck = model.bottleneck
    box_head = model.box_head
    torch_model = VT(backbone, bottleneck, box_head, head_type=cfg.MODEL.HEAD_TYPE, neck_type=cfg.MODEL.NECK.TYPE)
    torch_model.cuda()
    torch_model.eval()
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    sz_z = cfg.TEST.TEMPLATE_SIZE
    search = get_data(bs=bs, sz=sz_x)
    search_cuda = search.cuda()
    template = get_data(bs=bs, sz=sz_z)
    template_cuda = template.cuda()
    torch.onnx.export(torch_model,  # model being run
                      (search_cuda, template_cuda),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['search', 'template'],  # model's input names
                      output_names=['outputs_coord_new'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1000
    # """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name)
    print("execuation providers:")
    print(ort_session.get_providers())
    # # compute ONNX Runtime output prediction
    ort_inputs = {'search': to_numpy(search).astype(np.float32),
                  'template': to_numpy(template).astype(np.float32)
                  }
    """warmup (the first one running latency is quite large for the onnx model)"""
    for i in range(50):
        # pytorch inference
        torch_outs = torch_model(search_cuda, template_cuda)
        # onnx inference
        ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    t_pyt = 0  # pytorch time
    t_ort = 0  # onnxruntime time
    s_pyt = time.time()
    for i in range(N):
        torch_outs = torch_model(search_cuda, template_cuda)
    e_pyt = time.time()
    lat_pyt = e_pyt - s_pyt
    t_pyt += lat_pyt
    s_ort = time.time()
    for i in range(N):
        ort_outs = ort_session.run(None, ort_inputs)
    e_ort = time.time()
    lat_ort = e_ort - s_ort
    t_ort += lat_ort
    print("pytorch model average latency", t_pyt/N*1000)
    print("onnx model average latency:", t_ort/N*1000)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(torch_outs[0]) - ort_outs[0]))))
    #



