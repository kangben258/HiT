import os
root_path = os.path.dirname(__file__)+"/../../.."
val_path = os.path.join(root_path,"data/got10k/val")
if not os.path.exists(val_path):
    os.system("mkdir -p %s" % (val_path))
val_list_path_s = os.path.join(root_path,"lib/train/data_specs/got10k_vot_val_split.txt")
val_list_path_d = os.path.join(val_path,"list.txt")
os.system("cp %s %s" % (val_list_path_s, val_list_path_d))
f = open(val_list_path_d)
for index in f.readlines():
    index = index.split('\n')[0]
    t_name = index
    index = index.zfill(6)
    s_name = "GOT-10k_Train_"+index
    s = os.path.join(root_path,"data/got10k",s_name)
    t = os.path.join(val_path,t_name)
    # os.system("cp -r %s %s" % (s, t))
    os.system("ln -s %s %s" % (s, t))