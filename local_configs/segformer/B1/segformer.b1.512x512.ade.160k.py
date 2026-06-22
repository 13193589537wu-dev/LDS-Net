_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/sw.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
# custom_imports = dict(
#     imports=['mmseg.models.losses.dice_loss'],  # 或你自己的路径
#     allow_failed_imports=False
# )
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
# find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=None,     #'pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        # loss_decode=[
        #     dict(type='CrossEntropyLoss',  use_sigmoid=False, loss_weight=1.0),
        #     dict(type='DiceLoss',use_sigmoid=False,loss_weight=3.0,ignore_index=255)
        # ]),
        # loss_decode=[
        # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[0.5, 3.0]),
        # dict(type='DiceLoss', use_sigmoid=False, loss_weight=3.0, ignore_index=255)
        # ]),
        # decode_head loss 配置
        loss_decode = dict(
        type='MultiLoss',
        use_sigmoid=False,   # 给 DiceLoss 使用
        losses=[
            dict(
                type='CrossEntropyLoss',
                loss_weight=1.0
                # class_weight=[0.5, 3.0],
            )
            # dict(
            #     type='DiceLoss',
            #     loss_weight=3.0,
            #     ignore_index=255
            # )
        ]
    )),
        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[0.5, 3.0])),
        # loss_decode=dict(
        #     type='DiceLoss',
        #     use_sigmoid=False,
        #     loss_weight=3.0,
        #     ignore_index=255
        # )),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
work_dir='./work_dirs/qtpl/segformer'
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)