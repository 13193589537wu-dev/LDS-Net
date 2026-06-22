_base_ = [
    '../_base_/datasets/sw.py',  # 数据集配置
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

# BN 更稳定，建议保留
norm_cfg = dict(type='BN', requires_grad=True)

# 图像归一化配置（可根据数据集调整）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 模型配置
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='WeakMedSAM',
        in_channels=3,
        embed_dims=[128, 256, 512, 768],  # 四个 stage 的通道数
        depths=[2, 2, 6, 2],              # 每个 stage 的 Transformer block 数量
        num_heads=[4, 8, 16, 16],         # 每个 stage 的注意力头
        mlp_ratio=4.,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,  # 只使用最后一个 stage 的输出通道（整数！）
        channels=256,     # 中间通道数
        num_convs=1,      # 卷积层数（减少以节省显存）
        concat_input=True,  # 拼接输入特征（标准 FCN 设置）
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',      # 保持你的自定义损失
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
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器（保持不变）
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

# 学习率策略（保持不变）
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# 评价指标（保持不变）
evaluation = dict(
    interval=3000,
    metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
)
test_evaluator = dict(
    type='IoUMetric',
    metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
)

# 工作目录（保持不变）
work_dir = './work_dirs/qtpl/weakmedsam'

# 模型保存策略（保持不变）
checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
fp16 = dict(loss_scale=512.)
# _base_ = [         # 需要你准备 WeakMedSAM 基础模型定义
#     '../_base_/datasets/ade20k_repeat.py', # 数据集配置
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_160k_adamw.py'
# ]

# # BN 更稳定，建议保留
# norm_cfg = dict(type='BN', requires_grad=True)

# # 图像归一化配置（可根据数据集调整）
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=True
# )

# # 模型配置
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(
#         type='WeakMedSAM',
#         in_channels=3,
#         embed_dims=[128, 256, 512, 768],  # 四个 stage 的通道数
#         depths=[2, 2, 6, 2],              # 每个 stage 的 Transformer block 数量
#         num_heads=[4, 8, 16, 16],         # 每个 stage 的注意力头
#         mlp_ratio=4.,
#         drop_rate=0.1,
#         attn_drop_rate=0.1
#     ),
#     decode_head=dict(
#         type='FCNHead',
#         in_channels=[128, 256, 512, 768],  # 对齐
#         in_index=[0, 1, 2, 3],
#         feature_strides=[4, 8, 16, 32],
#         channels=256,
#         dropout_ratio=0.1,
#         num_classes=2,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         align_corners=False,
#         decoder_params=dict(
#             embed_dim=256,   # 统一 decode head 内部维度
#             mlp_ratio=4.0,
#             drop_rate=0.,
#             drop_path_rate=0.,
#         ),
#         loss_decode=dict(
#             type='MultiLoss',      # 老版本的写法
#             use_sigmoid=False,     # 给 DiceLoss 使用
#             losses=[
#                 dict(
#                     type='CrossEntropyLoss',
#                     loss_weight=1.0,
#                     class_weight=[0.5, 3.0]
#                 ),
#                 dict(
#                     type='DiceLoss',
#                     loss_weight=3.0,
#                     ignore_index=255
#                 )
#             ]
#         )
#     ),
#     auxiliary_head=None,
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole')
# )
# # 优化器
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }
#     )
# )

# # 学习率策略
# lr_config = dict(
#     policy='poly',
#     warmup='linear',
#     warmup_iters=5000,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False
# )

# # 评价指标
# evaluation = dict(
#     interval=3000,
#     metric=['mIoU', 'mDice', 'mRecall', 'mPrecision', 'aAcc', 'mFscore']
# )
# test_evaluator = dict(
#     type='IoUMetric',
#     metrics=['mIoU', 'mDice', 'mPrecision', 'mRecall', 'aAcc', 'mFscore']
# )

# # 工作目录
# work_dir = './ditch_work/ditch_work_weakmedsam'

# # 模型保存策略
# checkpoint_config = dict(interval=3000, max_keep_ckpts=4)
# fp16 = dict(loss_scale=512.)