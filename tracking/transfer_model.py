import torch
import collections

# path_old = '/home/cx/TransT/models/DETR_ep0397.pth.tar'
# path_new = '/home/cx/TransT/models/transt_N2.pth'
# # path_1 = '/home/cx/TransT/transt_para.pth'
# # path_2 = '/home/cx/TransT/transt_1.pth'
# # path_3 = '/home/cx/TransT/atom_default.pth'
#
# old = torch.load(path_old)
# new = torch.load(path_new)
#
# old_net = old['net']
# new_net = new['net']
#
# old2new = {}
# old2new_net = collections.OrderedDict()
#
# for key in old_net.keys():
#     if key == 'transformer.decoder.layers.0.norm2.weight':
#         new_key = 'featurefusion_network.decoder.layers.0.norm1.weight'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm2.bias':
#         new_key = 'featurefusion_network.decoder.layers.0.norm1.bias'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm3.weight':
#         new_key = 'featurefusion_network.decoder.layers.0.norm2.weight'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm3.bias':
#         new_key = 'featurefusion_network.decoder.layers.0.norm2.bias'
#         old2new_net[new_key] = old_net[key]
#     elif key[0:11] == 'transformer':
#         new_key = 'featurefusion_network' + key[11:]
#         old2new_net[new_key] = old_net[key]
#     else:
#         old2new_net[key] = old_net[key]
# old2new['net'] = old2new_net
# old2new['constructor'] = new['constructor']
# torch.save(old2new, '/home/cx/TransT/models/transt_N2_got10k_397.pth')

# path = '/home/cx/cx1/trdimp_net.pth.tar'
# model = torch.load(path)
# net = model['net']
# torch.save(net, '/home/cx/cx1/trdimp.pth')

# path1 = '/home/cx/cx1/TransT_experiments/submit/check1/TransT_M-code-2021-05-31T14_53_03.828056/TransT_M/models/TransTiouhsegm_ep0090.pth.tar'
# path2 = '/home/cx/cx1/work_dir/work_dir_mt_2t_e464_iouh_loss/checkpoints/ltr/transt/transt_iouh_ddp/TransTiouh_ep0090.pth.tar'
# model1 = torch.load(path1)
# model2 = torch.load(path2)
# net1 = model1['net']
# net2 = model2['net']
# net3 = net1
# for key in net3.keys():
#     if key in net2.keys():
#         net3[key] = net2[key]
# model3 = {}
# net3 = collections.OrderedDict()
# for key in net1.keys():
#     if key[0:6] == 'transt':
#         net3[key[7:]] = net1[key]
#     else:
#         net3[key] = net1[key]
path1 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/mtr/v_b_16_256_bs128_500k_e200_nl_lopre/MTR_ep0200.pth.tar'
model1 = torch.load(path1)
# path2 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vt/v_l_16_384_fb_bs16/VT_ep0020.pth.tar'
# model2 = torch.load(path2)
net1 = model1['net']
path2 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/mtr/v_b_16_256_nl/MTR_ep0010.pth.tar'
model2 = torch.load(path2)
# path2 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vt/v_l_16_384_fb_bs16/VT_ep0020.pth.tar'
# model2 = torch.load(path2)
net2 = model2['net']
a=1

# net2 = model2['net']
# path3 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/debug/x.pth'
# input1 = torch.load(path3)
# path4 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/debug/x_3090.pth'
# input2 = torch.load(path4)
# for key in net2.keys():
#     a = ((net1[key] == net2[key]) + 0).min()
#     if a == 0:
#         print(key)

# path3 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vtmt2/v_l_16_256/VTMT_ep0050.pth.tar'
# model3 = torch.load(path3)
# model3 = {'net':collections.OrderedDict()}
# model3['net'].update(model1['net'])
# model3['net'].update(model2['net'])
# torch.save(model3, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vtmt2/v_l_16_256_bs16/VTMT_ep0050.pth.tar')

# constructor1 = model1['constructor']
# model2 = {}
# model2['net'] = net1
# model2['constructor'] = constructor1
# torch.save(model2, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vittrack_baseline/vit_l_16_256_bs16/model.pth')
#

