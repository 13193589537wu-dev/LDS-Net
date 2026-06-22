_base_ = [
    '../_base_/datasets/sw.py',       # 保持数据集一致
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 1. 基础配置同步 (与 UNet 保持一致)
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
        type='FFNetResNet',            # 使用我们适配的 Backbone
        depth=50,                      # 对应常用的 FFNet-50 级别
        num_stages=4,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='FFNetHead',              # 使用我们适配的 FFNetHead
        in_channels=[256, 512, 1024, 2048], # ResNet-50 默认输出通道
        in_index=[0, 1, 2, 3],
        head_type='A',                 # 对应 AAA 系列的高精度配置
        channels=512,
        num_classes=2,                 # 同步类别数
        norm_cfg=norm_cfg,
        align_corners=False,           # 与 UNet 保持一致
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')        # 全图预测
)

# 2. 优化器同步 (与 UNet 100% 对齐)
optimizer = dict( 
    type='AdamW', 
    lr=0.0001, 
    betas=(0.9, 0.999), 
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)   # 遵循你 UNet 中的头部学习率倍增策略
        })
)

# 3. 学习率调度同步
lr_config = dict( 
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

# 4. 评价指标同步
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

# 重要：更改工作目录，避免覆盖 UNet 结果
work_dir='./work_dirs/qtpl/ffnet'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)