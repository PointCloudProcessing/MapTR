# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTR with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8
```
```
export CUDA_VISIBLE_DEVICES=0,1 
echo $CUDA_VISIBLE_DEVICES
nohup python3 -m torch.distributed.launch --nproc_per_node=1 tools/train.py projects/configs/maptr/maptr_tiny_r50_24e_t4.py --launcher pytorch ${@:3} --deterministic > output_partdata_train.txt &
```
```
export CUDA_VISIBLE_DEVICES=1 
echo $CUDA_VISIBLE_DEVICES

nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 tools/train.py projects/configs/maptr/maptr_tiny_fusion_24e.py --launcher pytorch ${@:3} --deterministic > output_partdata_front_train_numcam1.txt &

nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 tools/train.py projects/configs/maptr/maptr_tiny_fusion_24e_numcams_1.py --launcher pytorch ${@:3} --deterministic > output_partdata_front_train_numcam1_10000.txt &

```
```
export CUDA_VISIBLE_DEVICES=0

nohup python3 -m torch.distributed.launch --nproc_per_node=1 tools/train.py projects/configs/maptr/maptr_tiny_fusion_24e_numcams1_1w_1.py --launcher pytorch ${@:3} --deterministic > maptr_tiny_fusion_24e_numcams1_1w_1.txt &
```
```
export CUDA_VISIBLE_DEVICES=1 
nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 tools/train.py projects/configs/maptr/maptr_tiny_fusion_24e_numcams_full1.py --launcher pytorch ${@:3} --deterministic > output_partdata_front_train_numcam1_1_full.txt &
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_fusion_24e_numcams_1.py work_dirs/maptr_tiny_fusion_24e_numcams_1/epoch_1.pth 1
```

```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 tools/test.py projects/configs/maptr/maptr_tiny_fusion_24e_numcams_1_ld.py work_dirs/maptr_tiny_fusion_24e_numcams_1/epoch_1.pth --launcher pytorch ${@:4} --eval chamfer
```


# Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`