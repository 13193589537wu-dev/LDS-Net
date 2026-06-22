_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 归一化配置
norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 模型配置
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MSCAN',              # 经典 SegNeXt backbone
        arch='s',
        in_chans=3,
        depths=[3, 3, 9, 3],       # small 配置
        out_indices=(0, 1, 2, 3),
        init_cfg=None
    ),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,   # 按数据集修改
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',
            use_sigmoid=False,
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
        )
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        }
    )
)

# 学习率调度
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
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc','mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore']
)
# checkpoint 配置
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)

# 工作目录
work_dir = './ditch_work/ditch_work_segnext'
