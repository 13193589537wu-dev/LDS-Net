import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..builder import HEADS
from mmcv.cnn import ConvModule
from mmseg.models.losses import accuracy

@HEADS.register_module()
class UNetHead(BaseDecodeHead):
    """UNet decode head for MMSeg 0.13"""

    def __init__(self,
                 num_convs=2,
                 concat_input=True,
                 upsample_cfg=dict(type='InterpConv', scale_factor=2, mode='bilinear', align_corners=False),
                 **kwargs):
        super(UNetHead, self).__init__(**kwargs)

        self.num_convs = num_convs
        self.concat_input = concat_input
        self.upsample_cfg = upsample_cfg

        # 卷积堆
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=self.in_channels if i == 0 else self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.convs = nn.Sequential(*convs)

        # concat_input 的卷积
        if self.concat_input:
            self.conv_cat = ConvModule(
                in_channels=self.in_channels + self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        out = self.convs(x)

        if self.concat_input:
            out = self.conv_cat(torch.cat([x, out], dim=1))

        if self.upsample_cfg is not None:
            out = F.interpolate(out,
                                scale_factor=self.upsample_cfg.get('scale_factor', 2),
                                mode=self.upsample_cfg.get('mode', 'bilinear'),
                                align_corners=self.upsample_cfg.get('align_corners', False))

        out = self.cls_seg(out)
        return out

    # def losses(self, seg_logit, seg_label):
    #     """Compute segmentation loss."""
    #     loss = dict()
    #     seg_label = seg_label.squeeze(1)
    #     loss['loss_ce'] = self.loss_decode[0](seg_logit, seg_label)
    #     if len(self.loss_decode) > 1:  # 如果有 DiceLoss
    #         loss['loss_dice'] = self.loss_decode[1](seg_logit, seg_label)
    #     loss['acc_seg'] = accuracy(seg_logit, seg_label)
    #     return loss