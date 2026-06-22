_base_ = [
    '../_base_/models/fcn_r50-d8.py',        # 使用 ResNet + FCN 作为基础
    '../_base_/datasets/sw.py',   # 替换为你的数据集配置
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
    pretrained=None,      #'open-mmlab://resnet50_v1c',  # ImageNet 预训练
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),      # 空洞卷积设置
        strides=(1, 2, 1, 1),        # 下采样
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
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
                    loss_weight=1.0,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss'
            # loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    # paramwise_cfg=dict(custom_keys={
    #     # 'pos_block': dict(decay_mult=0.),
    #     'norm': dict(decay_mult=0.),
    #     'decode_head': dict(lr_mult=10.)
    # })
)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0, min_lr=0.0, by_epoch=False
)

checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

work_dir = './work_dirs/qtpl/resnet50'




