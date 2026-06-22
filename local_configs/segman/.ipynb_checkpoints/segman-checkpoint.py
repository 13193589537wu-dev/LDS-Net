_base_ = [
    '../../_base_/datasets/sw.py',  # 必须与 UNet 保持一致
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True) # 对比实验建议用同种 BN

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SegMANEncoder_b',      # 明确指定版本，对应下面的通道数
        pretrained=None),
    decode_head=dict(
        type='SegMANDecoder', 
        in_channels=[96, 160, 364, 560], # 对应 SegMAN_b 版本的 embed_dims
        in_index=[0, 1, 2, 3],
        channels=180,
        feat_proj_dim=320,
        dropout_ratio=0.1,
        num_classes=2,               # 必须改为 2，与 UNet 一致
        norm_cfg=norm_cfg,
        align_corners=False,
        # 建议 Loss 也写成和 UNet 一样的结构，方便严格对比
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# --- 以下部分与 UNet 配置保持 100% 绝对一致 ---

# optimizer
optimizer = dict( type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                  paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                  'norm': dict(decay_mult=0.),
                                                  'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict( policy='poly',
                  warmup='linear',
                  warmup_iters=5000,
                  warmup_ratio=1e-6,
                  power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc','mFscore']
)

test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc','mFscore']
)

# 路径必须分开！
work_dir='./work_dirs/sw/segman' 
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)