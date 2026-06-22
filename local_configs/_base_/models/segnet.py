norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,  # 如需加载 VGG 预训练权重，填入路径或 open-mmlab URI
    backbone=dict(
        type='VGG',
        depth=16,
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        dilations=(1, 1, 1, 1, 1),
        with_bn=True,
        ceil_mode=True,
        in_channels=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU')
    ),
    # 此处假设你已在代码里实现了 SegNetHead（存放路径：mmseg/models/decode_heads/segnet_head.py）
    decode_head=dict(
        type='SegNetHead',     # 请确保该类在代码中可导入（mmseg 会通过 registry 查找）
        in_channels=512,      # VGG16 最后 stage 通道数
        channels=512,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        dropout_ratio=0.1,
        loss_decode=dict(
            type='MultiLoss',
            use_sigmoid=False,
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
