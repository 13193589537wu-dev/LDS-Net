_base_ = [      # 如果有 UNet++ 模型可以替换
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
    pretrained=None,
    backbone=dict(
        type='NestedUNet',        # UNet++ backbone
        in_channels=3,
        base_channels=64,          # 起始通道数
        num_stages=5,              # 下采样阶段
        deep_supervision=True,     # U-Net++ 常用深监督
        strides=(1, 2, 2, 2, 2),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        norm_eval=False
    ),
    decode_head=dict(
        type='UNetPlusPlusHead',    # U-Net++ decode head
        in_channels=64,
        channels=64,
        num_classes=2,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
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
    auxiliary_head=None,             # 可选深监督辅助头
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

work_dir = './ditch_work/ditch_work_unet++'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
