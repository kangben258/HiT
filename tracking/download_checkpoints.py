import os
import _init_paths
root_path = os.path.join(os.path.dirname(__file__),"..")


def download_single_checkpoint(blob_name, project_dir, download_path, epoch):
    os.system("azcopy_linux_amd64_10.13.0/azcopy copy \"https://%s.blob.core.windows.net/v-zhaojie/%s/VTM_ep%04d.pth.tar?sv=2020-10-02&st=2022-04-25T05%%3A43%%3A10Z&se=2023-04-26T05%%3A43%%3A00Z&sr=c&sp=rl&sig=E4SKi%%2FHSsfoX7q6n9wG28NSQcB9KqnYMUe%%2B21ZsnpK0%%3D\" %s" %(blob_name, project_dir, epoch, download_path))


# download_path = os.path.join(root_path, "checkpoints/train/vt/v_l_16_256_tr4_bs16")
# project_dir = "projects/playground/playground/checkpoints/train/vt/v_l_16_256_tr4_bs16"
# # project_dir = "projects/exp_cx/checkpoints/train/vittrack_baseline/baseline_256"
# # for epoch in range(400,500,10):
# #     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
# for epoch in range(500,501,1):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)

download_path = os.path.join(root_path, "checkpoints/train/vtm2/v_b_16_256_tr4_bs16")
project_dir = "projects/playground/playground/checkpoints/train/vtm2/v_b_16_256_tr4_bs16"
# project_dir = "projects/exp_cx/checkpoints/train/vittrack_baseline/baseline_256"
# for epoch in range(400,500,10):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
for epoch in range(50,51,1):
    download_single_checkpoint("azsusw3", project_dir, download_path, epoch)

# download_path = os.path.join(root_path, "checkpoints/train/vtv/sv_l_16_256_concat_bs16")
# project_dir = "projects/playground/playground/checkpoints/train/vtv/sv_l_16_256_concat_bs16"
# # project_dir = "projects/exp_cx/checkpoints/train/vittrack_baseline/baseline_256"
# # for epoch in range(400,500,10):
# #     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
# for epoch in range(500,501,1):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)

# download_path = os.path.join(root_path, "checkpoints/train/vtv/vmv_l_16_256_k400_v2_bs16")
# project_dir = "projects/playground/playground/checkpoints/train/vtv/vmv_l_16_256_k400_v2_bs16"
# # project_dir = "projects/exp_cx/checkpoints/train/vittrack_baseline/baseline_256"
# # for epoch in range(400,500,10):
# #     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
# for epoch in range(500,501,1):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
#
# download_path = os.path.join(root_path, "checkpoints/train/vt/v_b_16_256_mp1_bs16")
# project_dir = "projects/playground/playground/checkpoints/train/vt/v_b_16_256_mp1_bs16"
# # project_dir = "projects/exp_cx/checkpoints/train/vittrack_baseline/baseline_256"
# # for epoch in range(400,500,10):
# #     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
# for epoch in range(500,501,1):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)