import argparse
import torch
from thop import profile
from thop.utils import clever_format
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
from lib.test.evaluation.environment import env_settings
from lib.models.vt.levit_ori import LeViT_384


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz=256):
    # img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True)
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    # update cfg
    args = parse_args()
    # build the model
    # transfer to test mode
    model = LeViT_384(fuse=False, pretrained=True)
    model.cuda()
    model.eval()
    # print(torch_model)
    # torch.save(torch_model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    image = get_data(bs=bs, sz=224)
    image_cuda = image.cuda()

    macs1, params1 = profile(model, inputs=[image_cuda], verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    save_name =  'levit.onnx'
    torch.onnx.export(model,  # model being run
                      (image_cuda),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # model's input names
                      output_names=['output'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1000
    # torch_model = torch_model.cuda()
    # torch_model.eval() # to move attention.ab to cuda for levit
    # torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    # torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()

    # """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name)
    # # ort_session.set_providers(["TensorrtExecutionProvider"],
    # #                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
    print("execuation providers:")
    print(ort_session.get_providers())
    # # compute ONNX Runtime output prediction
    # generate data
    # search = get_data(bs=bs, sz=sz_x)
    # template = get_data(bs=bs, sz=sz_z)
    # pytorch inference
    # search_cuda, template_cuda = search.cuda(), template.cuda()
    ort_inputs = {'input': to_numpy(image_cuda).astype(np.float32)
                  }
    """warmup (the first one running latency is quite large for the onnx model)"""
    for i in range(10):
        # pytorch inference
        torch_outs = model(image_cuda)
        # onnx inference
        ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    t_pyt = 0  # pytorch time
    t_ort = 0  # onnxruntime time
    s_pyt = time.time()
    for i in range(N):
        torch_outs = model(image_cuda)
    e_pyt = time.time()
    lat_pyt = e_pyt - s_pyt
    t_pyt += lat_pyt
    s_ort = time.time()
    for i in range(N):
        # ort_inputs = model(xz=model([search_cuda, template_cuda], mode="backbone"), mode="transformer")[0]
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
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")


