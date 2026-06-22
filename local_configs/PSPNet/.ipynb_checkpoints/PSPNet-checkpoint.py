_base_ = [
    # '../_base_/models/pspnet_r50-d8.py',   # 可以换 backbone，比如 resnet50-d8
    '../_base_/datasets/sw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,  # 可填预训练路径或 open-mmlab URI
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # 输出四个 stage
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,   # 对应 backbone 最后一层输出
        in_index=3,         # 使用最后一个 stage 输出
        channels=512,
        pool_scales=(1, 2, 3, 6),
        num_classes=2,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',
            use_sigmoid=False,
            losses=[
                dict(type='CrossEntropyLoss', loss_weight=1.0)
                     # , class_weight=[1.0, 1.0]),
                # dict(type='DiceLoss', loss_weight=3.0, ignore_index=255)
            ]
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,  # 对应 backbone 倒数第二层输出
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1, class_weight=[1.0, 1.0]
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)

test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

work_dir = './work_dirs/qtpl/pspnet'

checkpoint_config = dict(interval=3000, max_keep_ckpts=4)

# 可选 FP16 混合精度
fp16 = dict(loss_scale='dynamic')
