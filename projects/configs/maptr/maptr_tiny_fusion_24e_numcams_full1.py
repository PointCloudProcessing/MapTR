_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
# list of multiple file paths, it means inheriting from multiple files
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
point_cloud_range = [-15.0, -0.0, -2.0, 15.0, 30.0, 2.0]
lidar_point_cloud_range = [-15.0, -0.0, -5.0, 15.0, 30.0, 3.0]
voxel_size = [0.1, 0.1, 0.2]

num_cams = 1


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# map_classes = ['boundary']

# fixed_ptsnum_per_line = 20
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    modality='fusion',
    lidar_encoder=dict(
        voxelize=dict(max_num_points=10,point_cloud_range=lidar_point_cloud_range,voxel_size=voxel_size,max_voxels=[90000, 120000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=5,
            sparse_shape=[300, 600, 41],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                        128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock'
        ),
    ),
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # If used as backbone, list of indices of features to output. -1 means the last feature. 3 means the output of the third stage.
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],  # If used as neck, list of input channel counts at each stage of the backbone. Resnet18 512 Resnet50 2048
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=50,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapTRPerceptionTransformer',
            num_cams = num_cams,
            use_cams_embeds=False,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            modality='fusion',
            fuser=dict(
                type='ConvFuser',
                in_channels=[_dim_, 256],
                out_channels=_dim_,
            ),
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams=num_cams,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4, # 输出特征图的大小与输入点云的比例
        assigner=dict(  # 目标分配器的类型和相关参数，将点云目标检测中的点云目标分配给锚点
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesLocalMapDataset'
data_root = '/media/NAS/raw_data/ShuoShen/nuscenes/train/nuscenes/'
file_client_args = dict(backend='disk')

reduce_beams=32
load_dim=5
use_dim=5

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'points'])
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             pc_range=point_cloud_range,
             fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
             eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
             padding_value=-10000,
             map_classes=map_classes,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
              eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
              padding_value=-10000,
              map_classes=map_classes,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,  
    # This sets the initial learning rate for the optimizer to 6e-4. The learning rate determines the step size during gradient descent 
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.05), 
            # the learning rate is multiplied by 0.1, meaning it will be 10 times smaller than the base learning rate (6e-4)
        }),
    weight_decay=0.05,
    eps = 1e-8) 
# controls the L2 regularization strength (weight decay). It applies a penalty to the model's weights to discourage them from becoming too large

optimizer_config = dict(grad_clip= 
                        # which is a technique used to prevent gradients from becoming too large and potentially causing numerical instability 
                        dict(max_norm=75,  
                             # This sets the maximum L2 norm of gradients to 35. If any gradient's L2 norm exceeds this value, it will be scaled down to ensure it does not exceed this threshold.
                                       norm_type=2))    # This specifies the type of norm
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100, 
    # 控制到哪一个iteration起开始试用指定lr  specifies the number of initial iterations during which the learning rate gradually increases from its initial value to the regular learning rate.
    warmup_ratio=1.0 / 5, 
    # specifies the ratio of the initial learning rate to the regular learning rate. A larger warmup_ratio means a longer warm-up phase, during which the learning rate increases gradually. This can be useful to ensure the model starts with a very low learning rate and spends a significant portion of training with smaller steps.
    min_lr_ratio=1e-8) # Setting a non-zero min_lr_ratio prevents the learning rate from becoming too small
total_epochs = 100
# total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=1, 
                  pipeline=test_pipeline, 
                  metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.) original
fp16 = dict(loss_scale=256.) 
checkpoint_config = dict(interval=1)
find_unused_parameters=True
