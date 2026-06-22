import torch
import torch.nn as nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@HEADS.register_module()
class UNetPlusPlusHead(BaseDecodeHead):
    """Classic U-Net++ decode head"""
    def __init__(self, in_channels=64, channels=64, num_classes=2, dropout_ratio=0.1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, dropout_ratio=dropout_ratio, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) if norm_cfg['type']=='BN' else nn.Identity()
        self.act1 = nn.ReLU(inplace=True) if act_cfg['type']=='ReLU' else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio>0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.cls_seg(x)
        return x
