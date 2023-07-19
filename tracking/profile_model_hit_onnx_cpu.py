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
import torch.onnx
import numpy as np
import onnx
import onnxruntime


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='HiT',
                        help='training script name')
    parser.add_argument('--config', type=str, default='HiT_Base', help='yaml configure file name')
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


def evaluate(ort_session, ort_inputs, bs):
    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    # overall
    for i in range(T_w):
        ort_outs = ort_session.run(None, ort_inputs)
    start = time.time()
    for i in range(T_t):
        ort_outs = ort_session.run(None, ort_inputs)
    end = time.time()
    avg_lat = (end - start) / (T_t * bs)
    print("The average overall latency is %.2f ms" % (avg_lat * 1000))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
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
    '''get some onnx model'''
    onnx_name = prj_path + "/checkpoints/train/%s/%s/VT_cpu_ep%04d.onnx" % (args.script, args.config, cfg.TEST.EPOCH)
    ort_session = onnxruntime.InferenceSession(onnx_name,providers=['CPUExecutionProvider'])
    if args.script == "HiT":
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        ort_inputs = {'search': to_numpy(search).astype(np.float32),
                      'template': to_numpy(template).astype(np.float32)
                      }
        # evaluate the model properties
        evaluate(ort_session, ort_inputs, bs=bs)
    else:
        raise NotImplementedError
