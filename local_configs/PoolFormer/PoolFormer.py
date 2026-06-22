_base_ = [
    '../_base_/datasets/sw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# norm
norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
model = dict(
    type='EncoderDecoder',
    pretrained=None,   # 不用预训练
    backbone=dict(
        type='PoolFormer',
        arch='s12',     # 经典小模型，s12/s24/s36/m36 都可以
        init_cfg=None
    ),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 128, 320, 512],   # 对应 PoolFormer-s12
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',
            use_sigmoid=False,
            losses=[
                dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0
                    # class_weight=[0.5, 3.0]
                )
                # dict(
                #     type='DiceLoss',
                #     loss_weight=3.0,
                #     ignore_index=255
                # )
            ]
        )
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


# optimizer
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

# lr schedule
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# evaluation
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)

# 单独 test_evaluator（你能支持）
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

# checkpoint
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)

# work dir
work_dir = './work_dirs/qtpl/poolformer'
