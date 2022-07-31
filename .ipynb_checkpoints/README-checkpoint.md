# EdgeTrack

##所有指令都在该工程根目录下执行
## Install the environment
**Option1**: Use the Anaconda
```
conda create -n playground python=3.8
conda activate playground
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install requirements.txt
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${STARK_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train EdgeTrack
--nproc_per_node代表GPU数

```
python -m torch.distributed.launch --nproc_per_node 8 lib/train/run_training.py --script vt --config lv_fpn_256_c384_3sch --save_dir .
```
我是用8个GPU训的，每个GPU上batch size 16，总共batch size是128，所以如果你用两个GPU训，需要把配置文件中的batchsize改为64，以保证总的batchsize不变，如果显存不够，总的batchsize64应当也可以，但最好还是保证总共128的batchsize

(Optionally) Debugging training with a single GPU
```
python lib/train/run_training.py --script vt --config lv_fpn_256_c384_3sch --save_dir .
```


## Test and evaluate on benchmarks
参数说明:

--threads 指定线程数，一般与GPU数量相等即可，调试时设为0

--num_gpus指定用来测试的GPU数

--debug代表调试层级，0代表正常测试，1代表测试时可视化图片但不保存结果，2代表测试时保存可视化的图片，也不保存结果，图片保存位置在tracker里指定

- LaSOT
```
python tracking/test.py vt lv_fpn_256_c384_3sch --dataset lasot --threads 2 --num_gpus 2 --debug 0
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py vt lv_fpn_256_c384_3sch --dataset got10k_test --threads 2 --num_gpus 2 --debug 0
python lib/test/utils/transform_got10k.py --tracker_name vt --cfg_name lv_fpn_256_c384_3sch
```
- TrackingNet
```
python tracking/test.py vt lv_fpn_256_c384_3sch --dataset trackingnet --threads 2 --num_gpus 2 --debug 0
python lib/test/utils/transform_trackingnet.py --tracker_name vt --cfg_name lv_fpn_256_c384_3sch
```
- VOT2022  
```
d external/vot22/vittrack_baseline
vot evaluate <name>
vot analysis <name> --nocache
```
- VOT2020-LT    这个我没测过，不一定好用
```
cd external/vot20_lt/<workspace_dir>
export PYTHONPATH=<path to the stark project>:$PYTHONPATH
bash exp.sh
```
## 在got10k_val上验证一个配置下的所有模型
```
python tracking/val.py vt lv_fpn_256_c384_3sch --dataset got10k_val --threads 2 --num_gpus 2
```

## 转换ONNX，测各种平台上的速度、flops、参数量。在CPU上测速时，需安装CPU版本的pytorch，以获得正常的速度，在pytorch官网找一下安装指令
```
python tracking/transfer_onnx
python tracking/profile_model_vt --script vt --config lv_fpn_256_c384_3sch
python tracking/profile_model_vt_cpu --script vt --config lv_fpn_256_c384_3sch
python tracking/profile_model_vt_onnx --script vt --config lv_fpn_256_c384_3sch
python tracking/profile_model_vt_onnx_cpu --script vt --config lv_fpn_256_c384_3sch
```

