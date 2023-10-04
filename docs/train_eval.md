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
nohup python3 -m torch.distributed.launch --nproc_per_node=1 tools/train.py projects/configs/maptr/maptr_tiny_r50_24e_t4.py --launcher pytorch ${@:3} --deterministic > output_fulldata_t4config.txt &
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py ./path/to/ckpts.pth 8
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`