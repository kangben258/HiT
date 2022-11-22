# HiT

## Install the environment
**Option1**: Use the Anaconda
```
conda create -n HIT python=3.8
conda activate HIT
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
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

## Train HiT
```
python -m torch.distributed.launch --nproc_per_node 8 lib/train/run_training.py --script vt --config HiT_Base --save_dir .
```
(Optionally) Debugging training with a single GPU
```
python lib/train/run_training.py --script vt --config HiT_Base --save_dir .
```

## Test and evaluate on benchmark
- LaSOT
```
python tracking/test.py vt HiT_Base --dataset lasot --threads 2 --num_gpus 2 --debug 0
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py vt HiT_Base --dataset got10k_test --threads 2 --num_gpus 2 --debug 0
python lib/test/utils/transform_got10k.py --tracker_name vt --cfg_name HiT_Base
```
- TrackingNet
```
python tracking/test.py vt HiT_Base --dataset trackingnet --threads 2 --num_gpus 2 --debug 0
python lib/test/utils/transform_trackingnet.py --tracker_name vt --cfg_name HiT_Base
```
- GOT10k-val
```
python tracking/val.py vt HiT_Base --dataset got10k_val --threads 2 --num_gpus 2
```
## Transformer onnx; test speed, flops, params
```
python tracking/transfer_onnx
python tracking/profile_model_vt --script vt --config HiT_Base
python tracking/profile_model_vt_cpu --script vt --config HiT_Base 
python tracking/profile_model_vt_onnx --script vt --config HiT_Base
python tracking/profile_model_vt_onnx_cpu --script vt --config HiT_Base
```

