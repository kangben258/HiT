# -*-coding:utf-8-*-
import sys
sys.path.append('/home/kb/EdgeTrack-main')
from lib.test.vot20.levit_vot22 import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('vt', 'lv_fpn_256_c384_3sch', vis=False)