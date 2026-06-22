_base_ = [
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
        type='VisionTransformer',
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=256,
        num_classes=2,
        embed_dims=768,
        num_heads=8,
        num_layers=2,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',      # 老版本的写法
            use_sigmoid=False,     # 给 DiceLoss 使用
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


optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9,0.999),
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
    metric=['mIoU','mDice','mRecall','mPrecision','aAcc','mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU','mDice','mPrecision','mRecall','aAcc','mFscore']
)

work_dir='./ditch_work/ditch_work_segmenter'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
