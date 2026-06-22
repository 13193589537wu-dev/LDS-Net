_base_ = [
    '../_base_/datasets/sw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 注意：去掉了 '../_base_/models/unet.py'，因为我们从头定义了完整的 MK_UNet 模型结构。

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
        type='MK_UNet',         # 采用标准大杯版本
        in_channels=3,
        num_classes=2           # 【重点】在这里告诉主干网络你要分割的类别数
    ),
    decode_head=dict(
        type='EmptyDecodeHead', # 使用我们自定义的空解码头
        in_channels=32,         # 占位符，不影响实际计算（MMSeg 底层需要）
        channels=32,            # 占位符，不影响实际计算
        num_classes=2,
        loss_decode=dict(       # 完全保留你原来的 MultiLoss 写法
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
    auxiliary_head=None,        # 不需要辅助头
    train_cfg=dict(),
    test_cfg=dict(mode='whole') # 保持全图预测
)

# optimizer 保持原样
optimizer = dict( type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict( policy='poly',
                 warmup='linear',
                 warmup_iters=5000, # 1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# evaluation / evaluator 保持原样
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc','mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore']
)

# 更改工作目录以防覆盖之前的实验
work_dir='./work_dirs/qtpl/mkunet'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)