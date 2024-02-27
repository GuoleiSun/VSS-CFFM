_base_ = [
    '../../_base_/models/segformer.py',
    # '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/cluster/work/cvl/celiuce/video-seg/models/segformer/pretrained_models/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),
    decode_head=dict(
        type='CFFMHead_clips_resize1_8_gene_prototype',
        # type='MLPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256, depths=2),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
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
    samples_per_gpu=2,
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
        split='train_val_generate_prototype',
        pipeline=test_pipeline,
        dilation=[]))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# data = dict(samples_per_gpu=2)
evaluation = dict(interval=160000, metric='mIoU')
