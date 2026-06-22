import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@HEADS.register_module()
class SegNetHeadLight(BaseDecodeHead):
    """
    轻量版 SegNetHead 替代实现
    使用 Upsample + Conv2d 上采样，节省显存
    """

    def __init__(self, in_channels, channels, num_classes, norm_cfg=None, dropout_ratio=0.1, **kwargs):
        super().__init__(in_channels=in_channels,
                         channels=channels,
                         num_classes=num_classes,
                         dropout_ratio=dropout_ratio,
                         norm_cfg=norm_cfg,
                         **kwargs)

        # decoder 阶段，逐步上采样
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//2, channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//4, channels//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels//8),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(channels//8, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[-1]  # 取 backbone 最深层输出
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.cls_seg(x)
        return x
