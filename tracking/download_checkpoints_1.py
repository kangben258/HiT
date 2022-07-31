import os
import _init_paths
root_path = os.path.join(os.path.dirname(__file__),"..")
download_path = os.path.join(root_path, "checkpoints/train/vittrack_baseline/vit_l_16_256_bs16")

def download_single_checkpoint(blob_name, project_dir, download_path, epoch):
    os.system("azcopy_linux_amd64_10.13.0/azcopy copy \"https://%s.blob.core.windows.net/v-zhaojie/%s/VITTRACK_BASELINE_ep%04d.pth.tar?sv=2020-10-02&st=2022-04-25T05%%3A43%%3A10Z&se=2023-04-26T05%%3A43%%3A00Z&sr=c&sp=rl&sig=E4SKi%%2FHSsfoX7q6n9wG28NSQcB9KqnYMUe%%2B21ZsnpK0%%3D\" %s" %(blob_name, project_dir, epoch, download_path))

project_dir = "projects/playground/l_16_256_bs16_8A100/checkpoints/train/vittrack_baseline/vit_l_16_256_bs16"
# for epoch in range(400,500,10):
#     download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
for epoch in range(500,501,1):
    download_single_checkpoint("azsusw3", project_dir, download_path, epoch)
