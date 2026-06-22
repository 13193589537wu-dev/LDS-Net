import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) Module for DeepLabV3+."""
    def __init__(self, in_channels, out_channels, dilations, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super(ASPPModule, self).__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1 if dilation == 1 else 3,
                    padding=0 if dilation == 1 else dilation,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

    def forward(self, x):
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        return outs


@HEADS.register_module()
class DeepLabV3PlusHead(BaseDecodeHead):
    """DeepLabV3+ Head.

    This head includes:
    - ASPP module
    - Image Pooling branch
    - Low-level feature fusion (from c1)
    """

    def __init__(self,
                 dilations=(1, 12, 24, 36),
                 c1_in_channels=256,
                 c1_channels=48,
                 **kwargs):
        super(DeepLabV3PlusHead, self).__init__(**kwargs)

        # ASPP
        self.aspp_modules = ASPPModule(
            self.in_channels,
            self.channels,
            dilations,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # Image Pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )

        # 1x1 conv to fuse ASPP + Image Pool
        self.bottleneck = ConvModule(
            len(dilations) * self.channels + self.channels,
            self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # Low-level feature projection (c1)
        self.c1_proj = ConvModule(
            c1_in_channels,
            c1_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # Final fuse layer
        self.fuse = nn.Sequential(
            ConvModule(
                self.channels + c1_channels,
                self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )

    def forward(self, inputs):
        # DeepLabV3+ uses high-level and low-level features
        x = self._transform_inputs(inputs)  # high-level feature
        c1 = inputs[0]  # low-level feature (stage1)

        # ASPP
        aspp_outs = self.aspp_modules(x)
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        aspp_outs.append(img_pool)

        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_outs = self.bottleneck(aspp_outs)

        # Upsample ASPP output to match c1 feature
        aspp_outs = F.interpolate(aspp_outs, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        # Project c1
        c1_out = self.c1_proj(c1)

        # Fuse ASPP + c1
        fused = torch.cat([aspp_outs, c1_out], dim=1)
        fused = self.fuse(fused)

        # Final prediction
        output = self.cls_seg(fused)
        return output
