# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTR with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8
```
Train MapTR2 with 2 GPUs 

```
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=28509 tools/train.py projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py --launcher pytorch ${@:3} --deterministic
```
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=28509 tools/train.py projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py --launcher pytorch ${@:3} --deterministic
```
kill all python program
```
sudo pkill -f python
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py ./path/to/ckpts.pth 8
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`