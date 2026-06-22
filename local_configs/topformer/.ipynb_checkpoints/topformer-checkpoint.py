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
        type='Topformer',
        # cfgs 参照 Topformer-Base 官方配置: [kernel_size, expand_ratio, channels, stride]
        cfgs=[
            [3, 1, 16, 1],
            [3, 4, 32, 2],
            [3, 3, 32, 1],
            [3, 3, 64, 2],
            [3, 3, 64, 1],
            [3, 3, 128, 2],
            [3, 3, 128, 1],
            [3, 3, 160, 2],
            [3, 3, 160, 1],
        ],
        channels=[32, 64, 128, 160],      # 各阶段通道数
        out_channels=[None, 64, 128, 160], # 注入后的输出通道数
        embed_out_indice=[2, 4, 6, 8],     # tpm 提取特征的层索引
        decode_out_indices=[1, 2, 3],      # 参与后续解码的阶段
        depths=4,
        key_dim=16,
        num_heads=8,
        attn_ratios=2,
        mlp_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        injection_type="muli_sum",
        injection=True),
    decode_head=dict(
        type='SimpleHead',                 # 使用上传的 simple_head.py 中的模块
        in_channels=[64, 128, 160],        # 对应 backbone 的 decode_out_indices 输出
        in_index=[0, 1, 2],
        channels=160,                      # embedding_dim
        num_classes=2,                     # 同步类别数
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
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
            'head': dict(lr_mult=10.)   # 遵循 UNet 中的头部学习率倍增策略
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

# 5. 实验路径管理
work_dir='./work_dirs/qtpl/topformer'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)