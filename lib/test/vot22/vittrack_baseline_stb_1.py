import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot22.vittrack_vot22_stb import run_vot_exp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_vot_exp('vt', 'cv_l_14_252_bs16', vis=False)
# run_vot_exp('vittrack_baseline', 'vit_l_16_384_bs16', vis=False)
