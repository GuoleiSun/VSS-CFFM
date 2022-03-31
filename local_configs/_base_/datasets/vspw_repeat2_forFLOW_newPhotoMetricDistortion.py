# dataset settings
dataset_type = 'VSPWDataset2'
data_root = 'data/vspw/VSPW_480p'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(853, 480), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips2'),
    dict(type='Normalize_clips2', **img_norm_cfg),
    dict(type='Pad_clips2', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips2'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(853, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            split='train',
            pipeline=train_pipeline,
            dilation=[-9,-6,-3])),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9,-6,-3]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9,-6,-3]))
