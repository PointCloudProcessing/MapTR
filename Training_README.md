# Errors
## cost invalid invalid numeric entries

```
len(cls_pred) 50
len(gt_labels) 6
Traceback (most recent call last):
  File "tools/train.py", line 260, in <module>
    main()
  File "tools/train.py", line 249, in main
    custom_train_model(
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/bevformer/apis/train.py", line 27, in custom_train_model
    custom_train_detector(
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py", line 199, in custom_train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 50, in train
    self.run_iter(data_batch, train_mode=True, **kwargs)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py", line 29, in run_iter
    outputs = self.model.train_step(data_batch, self.optimizer,
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/parallel/distributed.py", line 52, in train_step
    output = self.module.train_step(*inputs[0], **kwargs[0])
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmdet/models/detectors/base.py", line 237, in train_step
    losses = self(**data)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/detectors/maptr.py", line 163, in forward
    return self.forward_train(**kwargs)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 214, in new_func
    output = old_func(*new_args, **new_kwargs)
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/detectors/maptr.py", line 281, in forward_train
    losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/detectors/maptr.py", line 145, in forward_pts_train
    losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 214, in new_func
    output = old_func(*new_args, **new_kwargs)
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py", line 696, in loss
    losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmdet/core/utils/misc.py", line 29, in multi_apply
    return tuple(map(list, zip(*map_results)))
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py", line 520, in loss_single
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py", line 478, in get_targets
    pos_inds_list, neg_inds_list) = multi_apply(
  File "/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/mmdet/core/utils/misc.py", line 29, in multi_apply
    return tuple(map(list, zip(*map_results)))
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py", line 385, in _get_target_single
    assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
  File "/home/ld_t4/Documents/ShuoShen/MapTR/./projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py", line 192, in assign
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
ValueError: matrix contains invalid numeric entries
/media/NAS/raw_data/ShuoShen/miniconda_home/envs/maptr/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torch.distributed.run.
Note that --use_env is set by default in torch.distributed.run.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions
```


Reason:


```


学习率过高：
如果学习率设置得太高，可能会导致权重更新过大，使得损失函数值变得不稳定甚至出现NaN。解决方法是尝试降低学习率。
数值不稳定：
在某些情况下，数值不稳定可能会导致溢出或下溢，从而产生NaN值。为了避免数值不稳定，可以考虑使用数值稳定的版本的函数（例如使用log1p代替log）或者添加小的常数以避免除以零的情况。
参数初始化不当：
不合适的参数初始化可能会导致损失函数很快变成NaN。确保使用适当的初始化策略，例如Xavier或He初始化。
梯度爆炸：
如果模型的梯度变得非常大，权重更新可能会导致NaN值。可以尝试使用梯度裁剪（Gradient Clipping）来防止梯度爆炸。
模型结构问题：
检查模型结构，确保没有设计错误，例如错误的激活函数或损失函数的选择。
数据问题：
检查输入数据和标签，确保它们没有包含任何无效或NaN值。同时，检查数据的规模和范围，确保它们适合模型的需求。
硬件问题：
在极少数情况下，硬件问题（例如GPU故障）可能会导致NaN损失。如果怀疑是硬件问题，可以尝试在不同的硬件上运行模型，或者使用CPU而不是GPU来训练模型。
软件或库的bug：
确保使用的软件库和框架是最新的，并检查是否有已知的bug和修复程序。
为了诊断和修复问题，可以使用以下一些策略：
日志和监视：
记录训练过程中的所有重要指标，包括损失、梯度大小、权重更新等，以帮助诊断问题。
简化模型：
简化模型结构，从最基本的模型开始，然后逐渐增加复杂性，以找出导致问题的特定部分。
```

## If ros-melodic-cv-bridge is already installed but you're still encountering the ModuleNotFoundError for cv_bridge when running your Python script, it suggests that the Python environment you're using to run the script doesn't have access to the ROS libraries.
Source ROS:

Before running your script, ensure you've sourced the ROS environment for Melodic:

```
source /opt/ros/melodic/setup.bash
```



