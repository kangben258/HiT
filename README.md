* # [ICCV'2023] - Exploring Lightweight Hierarchical Vision Transformers for Efficient Visual Tracking

   > [**Exploring Lightweight Hierarchical Vision Transformers for Efficient Visual Tracking**]()<br>
   > accepted by ICCV2023<br>
   
   This is an official pytorch implementation of the ICCV2023 paper **Exploring Lightweight Hierarchical Vision Transformers for Efficient Visual Tracking**, a new framework for Efficient visual object tracking.
   
   
   
   ## Highlights
   
   ### Simple architecture 
   
   The architecture of HiT is very simple only contains three components: a lightweight hierarchical vision transformer, a bridge Module and a prediction head.
   
   ![HiT_Framework](/home/kb/kb/HiT/tracking/Framework.png)
   
   ### Strong performance
   
   HiT achieves promising speed with competitive performance, it is faster and performs better than previous efficient trackers, It significantly closes the gap between efficient trackers and mainstream trackers. It can be deployed in low-computing power scenarios.
   
   ![HiT_Framework](/home/kb/kb/HiT/tracking/compare.png)
   
   |                                           | HiT-Base | HiT-Small | HiT-Tiny | FEAR | TransT |
   | :---------------------------------------- | :------: | :-------: | :------: | :--: | :----: |
   | LaSOT (AUC)                               | **64.6** | **60.4**  | **54.7** | 53.5 |  64.9  |
   | GOT-10K (AO)                              | **64.0** | **62.6**  | **52.6** | 61.9 |  72.3  |
   | TrackingNet (AUC)                         | **80.0** | **77.7**  | **74.6** |  -   |  81.4  |
   | PyTorch Speed on 3090 GPU (FPS)           | **175**  |  **192**  | **204**  | 105  |   63   |
   | PyTorch Speed on i9-9900K CPU (FPS)       |  **33**  |  **72**   |  **76**  |  60  |   5    |
   | PyTorch Speed on Nvidia  Jetson AGX (FPS) |  **61**  |  **68**   |  **77**  |  38  |   13   |
   | PyTorch Speed on Nvidia  Jetson NX (FPS)  |  **32**  |  **34**   |  **39**  |  22  |   11   |
   | ONNX Speed on 3090 GPU (FPS)              | **274**  |  **400**  | **455**  |  -   |   -    |
   | ONNX Speed on i9-9900K CPU (FPS)          |  **42**  |  **98**   | **125**  |  -   |   -    |
   | ONNX Speed on Nvidia  Jetson AGX (FPS)    |  **75**  |  **119**  | **145**  |  -   |   -    |
   
   
   
   # Install the environment
   
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
   python -m torch.distributed.launch --nproc_per_node 8 lib/train/run_training.py --script HiT --config HiT_Base --save_dir .
   ```
   
   (Optionally) Debugging training with a single GPU
   
   ```
   python lib/train/run_training.py --script HiT --config HiT_Base --save_dir .
   ```
   
   ## Test and evaluate on benchmark
   
   - LaSOT
   
   ```
   python tracking/test.py HiT HiT_Base --dataset lasot --threads 2 --num_gpus 2 --debug 0
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```
   
   - GOT10K-test
   
   ```
   python tracking/test.py HiT HiT_Base --dataset got10k_test --threads 2 --num_gpus 2 --debug 0
   python lib/test/utils/transform_got10k.py --tracker_name HiT --cfg_name HiT_Base
   ```
   
   - TrackingNet
   
   ```
   python tracking/test.py HiT HiT_Base --dataset trackingnet --threads 2 --num_gpus 2 --debug 0
   python lib/test/utils/transform_trackingnet.py --tracker_name HiT --cfg_name HiT_Base
   ```
   
   - NFS
   
   ```
   python tracking/test.py HiT HiT_Base --dataset nfs --threads 2 --num_gpus 2 --debug 0
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```
   
   - UAV123
   
   ```
   python tracking/test.py HiT HiT_Base --dataset uav --threads 2 --num_gpus 2 --debug 0
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```
   
   - LaSOText
   
   ```
   python tracking/test.py HiT HiT_Base --dataset lasot_extension_subset --threads 2 --num_gpus 2 --debug 0
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```
   
   
   
   ## Run Video demo
   
   ```
   python tracking/video_demo.py <path of onnx model> <video path> 
   ```
   
   
   
   ## Transform onnx; test speed, flops, params
   
   ```
   python tracking/transfer_onnx
   python tracking/profile_model_hit --script HiT --config HiT_Base
   python tracking/profile_model_hit_cpu --script HiT --config HiT_Base 
   python tracking/profile_model_hit_onnx --script HiT --config HiT_Base
   python tracking/profile_model_hit_onnx_cpu --script HiT --config HiT_Base
   ```
   
   
   
   ## Models && Raw results
   
   The trained models, and the raw tracking results are provided in [here](https://drive.google.com/drive/folders/15VTIJnUtJTdU6TcmGOixSEcErYV-h_xL?usp=sharing)
   
   
   
   ## Acknowledgement
   
   * This codebase is implemented on [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking).We would like to thank their authors for providing great libraries.
   
     
   
   ## Contact
   
   * Ben Kang (email:kangben@mail.dlut.edu.cn)
   
   * Xin Chen (email:chenxin3131@mail.dlut.edu.cn)
   
     Feel free to contact if you have additional questions.
