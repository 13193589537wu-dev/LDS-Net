_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SPA',           # 你在 backbones 里写的 spa.py
        in_channels=3,
        base_channels=64
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=256,      # 对应 backbone.out_channels
        in_index=0,           # SPA 返回 (x,) tuple → index=0
        channels=128,
        pool_scales=(1, 2, 3, 6),   # PSPNet 经典 pyramid pooling
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',      # 老版本的写法
            use_sigmoid=False,     # 给 DiceLoss 使用
            losses=[
                dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0,
                    class_weight=[0.5, 3.0]
                ),
                dict(
                    type='DiceLoss',
                    loss_weight=3.0,
                    ignore_index=255
                )
        ]
    )),
    auxiliary_head=dict(       # PSPNet 经典有辅助头
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=64,
        num_convs=1,
        num_classes=2,
        norm_cfg=norm_cfg,
        concat_input=False,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc','mFscore']
)
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
work_dir='./ditch_work/ditch_work_spa'
