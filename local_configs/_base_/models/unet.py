norm_cfg = dict(type='BN', requires_grad=True)  # 原版UNet没有BN，但可以保留BN更稳定
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,  # 起始通道数64
        num_stages=5,  # 原论文UNet是5个下采样阶段
        strides=(1, 2, 2, 2, 2),  # 每个stage下采样，原始第一层不下采样
        enc_num_convs=(2, 2, 2, 2, 2),  # 每个stage两个卷积
        dec_num_convs=(2, 2, 2, 2),  # 对称的decoder
        downsamples=(True, True, True, True),  # 下采样标志
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,  # 是否checkpoint
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),  # 原版UNet用deconv，但InterpConv也OK
        norm_eval=False),
    decode_head=dict(
        type='UNetHead',  # 用UNetHead，而不是FCNHead，保持结构一致
        in_channels=64,  # 最后一层channel数
        channels=64,
        num_classes=2,
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,  # 经典UNet没有auxiliary head
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # 原版UNet用whole测试，不用slide
