import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import kaiming_init
from ..builder import BACKBONES


class SPABlock(nn.Module):
    """Spatial Pyramid Attention Module"""

    def __init__(self, in_channels, reduction=16, pool_sizes=(1, 2, 3, 6)):
        super(SPABlock, self).__init__()
        self.pool_sizes = pool_sizes
        self.reduction = reduction
        mid_channels = in_channels // reduction

        # 用于不同池化分支的卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
            for ps in pool_sizes
        ])

        # 聚合后生成注意力
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels * len(pool_sizes), in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = []
        for conv in self.convs:
            y = conv(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            out.append(y)
        out = torch.cat(out, dim=1)
        attn = self.attention(out)
        return x * attn


@BACKBONES.register_module()
class SPA(BaseModule):
    """SPA Backbone for segmentation"""

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_classes=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            SPABlock(base_channels)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            SPABlock(base_channels*2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            SPABlock(base_channels*4)
        )

        self.out_channels = base_channels*4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return (x,)   # mmseg 规范：返回 tuple
        
    def init_weights(self, pretrained=None):
        """Initialize weights for SPA backbone."""
        if pretrained is not None:
            # 如果你有预训练模型可以在这里加载
            self.load_state_dict(torch.load(pretrained), strict=False)
        else:
            # 没有预训练的话，就给卷积层随机初始化
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
