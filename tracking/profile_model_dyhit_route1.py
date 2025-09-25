import argparse
import torch
import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import _init_paths
from lib.utils.merge import merge_template_search
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import lib.models.HiT.levit_utils as utils
from torch.nn import functional as F

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='DyHiT',
                        help='training script name')
    parser.add_argument('--config', type=str, default='stage2', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])
class MLP(torch.nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = MLP(384,94,1,3)
    def forward(self,x):
        difct_score = self.router(xz1[:, :256, :]).sigmoid()
        select_element = difct_score[difct_score > 0.6]
        score = torch.mean(select_element).item()
        return  score

def evaluate(model, images_list,router,xz1):
    """Compute FLOPs, Params, and Speed"""
    '''Speed Test'''
    """Compute FLOPs, Params, and Speed"""
    # # backbone
    macs1, params1 = profile(router, inputs=(xz1), verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('router macs is ', macs)
    print('router params is ', params)
    # head
    macs2, params2 = profile(model, inputs=(images_list, True,False,0.9,0.9,False,0.6), verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('model macs is ', macs)
    print('model params is ', params)

    T_w = 10
    T_t = 100
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            out_dict,score = model(images_list=images_list,first_score=1,frame=False,threshold=-99999)
        start = time.time()
        for i in range(T_t):
            out_dict,score = model(images_list=images_list,first_score=1,frame=False,threshold=-99999)
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))

def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # device = "cpu"
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = prj_path + '/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    h_dim = cfg.MODEL.HIDDEN_DIM
    '''import vt network module'''
    model_module = importlib.import_module('lib.models.HiT')
    model_constructor = model_module.build_dyhit
    model = model_constructor(cfg)
    # merge conv+bn for levit
    utils.replace_batchnorm(model.model1.backbone.body)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    model.eval()
    router = router().to(device)
    # evaluate the model properties
    images_list = [search, template]
    xz1 = torch.randn((1,320,384)).to(device)
    # xz = model.forward_backbone(images_list,first_score=1,threshold=1)
    evaluate(model,images_list,router=router,xz1=xz1)
