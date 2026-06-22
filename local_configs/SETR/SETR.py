_base_ = [
    '../_base_/datasets/sw.py',  # 保持与 UNet 一致的数据集
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# 参数设置：基于源码中的 VIT_MLA 定义
norm_cfg = dict(type='SyncBN', requires_grad=True) # Transformer 推荐使用 SyncBN
img_size = 256 # 对应 VIT_MLAHead 的默认值，请确保与你的输入图像尺寸一致

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VIT_MLA',
        model_name='vit_large_patch16_384', # 源码默认模型名
        random_init=True,
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=2,      # 对应你的任务类别
        drop_rate=0.1,
        mla_channels=256,   # 源码默认 MLA 通道数
        mla_index=(5, 11, 17, 23), # 提取特征的层索引
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False
    ),
    decode_head=dict(
        type='VIT_MLAHead',
        img_size=img_size,
        in_channels=1024,      # 对应 backbone 的 embed_dim
        channels=256,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=2,      # 对应你的任务类别
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss', # 保持与你 UNet 配置一致的 Loss 结构
            use_sigmoid=False,
            losses=[
                dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0
                )
            ]
        )
    ),
    # SETR 原始论文通常带有辅助头，如果不需要可设为 None，保持与 UNet 对比简洁性
    auxiliary_head=None, 
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器配置：针对 Transformer 的学习率和权重衰减微调
optimizer = dict(
    type='AdamW',
    lr=0.0001, # Transformer 通常需要比 CNN 稍低的学习率，建议在 1e-4 到 6e-5 之间尝试
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)
# 学习率调度：保持 poly 策略
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# 评估设置
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore'],
    save_best='mIoU'
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore']
)
work_dir = './work_dirs/qtpl/setr'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)