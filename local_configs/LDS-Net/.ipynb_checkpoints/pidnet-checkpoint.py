_base_ = [
    '../_base_/datasets/sw.py',       # 保持数据集一致
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 1. 同步归一化和图像配置 (与 UNet 保持一致)
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
        type='PIDNetBackbone',         # 使用适配后的 PIDNet 主干
        m=2, n=3, 
        planes=32,                     # PIDNet-S 配置
        ppm_planes=96,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='PIDHead',                # 使用适配后的 PIDNet 解码头
        in_channels=[64, 128, 64, 64, 64], 
        in_index=[0, 1, 2, 3, 4],
        channels=128,
        num_classes=2,                 # 同步类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        # PIDNet 必须使用多头损失才能正常运行其三分支逻辑
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0), # 主损失
            dict(type='CrossEntropyLoss', loss_weight=0.4),      # P分支辅助
            dict(type='BoundaryLoss', loss_weight=1.0)     # D分支边界
        ]),
    auxiliary_head=None,               # PIDNet 逻辑已集成在 head 中
    train_cfg=dict(),
    test_cfg=dict(mode='whole')        # 保持全图推断一致
)

# 2. 优化器配置 (与 UNet 100% 对齐)
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
resume_from='./work_dirs/qtpl/pidnet/Overall/1/iter_75000.pth'
# 3. 学习率调度 (与 UNet 100% 对齐)
lr_config = dict( 
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

# 4. 评价指标与实验管理 (与 UNet 100% 对齐)
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

# 更改工作目录，区分对比实验结果
work_dir='./work_dirs/qtpl/pidnet/Overall/1'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)