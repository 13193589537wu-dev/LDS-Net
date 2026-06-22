_base_ = [
    '../_base_/models/unet.py',
    '../_base_/datasets/sw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]
norm_cfg = dict(type='BN', requires_grad=True)  # BN 更稳定，可保留
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,   # 起始通道数
        num_stages=5,       # UNet 5 个下采样阶段
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
        type='UNetHead',
        in_channels=64,
        channels=64,
        num_classes=2,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',      # 老版本的写法
            use_sigmoid=False,     # 给 DiceLoss 使用
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
    auxiliary_head=None,        # 经典 UNet 没有辅助头
    train_cfg=dict(),
    test_cfg=dict(mode='whole') # 全图预测
)
# optimizer
optimizer = dict( type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))   #lr=0.00001,_delete_=False,

lr_config = dict( policy='poly',
                 warmup='linear',
                 warmup_iters=5000,#1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)  #_delete_=True,

# resume_from='./work_dirs/latest.pth'
# data = dict(samples_per_gpu=2)  #原2
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc','mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore']
)
work_dir='./work_dirs/qtpl/unet'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)