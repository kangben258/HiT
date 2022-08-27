# -*-coding:utf-8-*-
import torch
import time
# checkpoint = torch.hub.load_state_dict_from_url(
#     'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth', map_location='cpu')
# state_dict = checkpoint['model']
# print('finish')
# state_dict_load = OrderedDict()
# for key in state_dict.keys():
#     if key[0:7] == 'visual.':
#         state_dict_load[key[7:]] = state_dict[key]
a = torch.rand(128,256,16,16)
print("a:",a)
b = a.view(128,256,-1)
print('b',b)
time_a_s = time.time()
c1 = a.view(128,64,4,256)
time_a_e = time.time()
time_a = time_a_e - time_a_s
print('time_a:',time_a)
time_b_s = time.time()
c2 = b.view(128,64,4,256)
time_b_e = time.time()
time_b = time_b_e - time_b_s
print('time_b:',time_b)


