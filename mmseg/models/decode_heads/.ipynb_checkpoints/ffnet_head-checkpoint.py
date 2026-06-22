import torch
import torch.nn as nn
import torch.nn.functional as F
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from mmcv.cnn import ConvModule

class UpBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 对应原代码中的特征融合逻辑
        self.fam_32_sm = ConvModule(in_channels[3], out_channels[3], 3, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'))
        self.fam_32_up = ConvModule(in_channels[3], in_channels[2], 1, padding=0, conv_cfg=None, norm_cfg=dict(type='BN'))
        
        self.fam_16_sm = ConvModule(in_channels[2], out_channels[2], 3, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'))
        self.fam_16_up = ConvModule(in_channels[2], in_channels[1], 1, padding=0, conv_cfg=None, norm_cfg=dict(type='BN'))
        
        self.fam_8_sm = ConvModule(in_channels[1], out_channels[1], 3, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'))
        self.fam_8_up = ConvModule(in_channels[1], in_channels[0], 1, padding=0, conv_cfg=None, norm_cfg=dict(type='BN'))
        
        self.fam_4 = ConvModule(in_channels[0], out_channels[0], 3, padding=1, conv_cfg=None, norm_cfg=dict(type='BN'))

    def forward(self, x):
        feat4, feat8, feat16, feat32 = x
        
        sm32 = self.fam_32_sm(feat32)
        up32 = F.interpolate(self.fam_32_up(feat32), size=feat16.shape[2:], mode='bilinear', align_corners=True)
        
        x16 = up32 + feat16
        sm16 = self.fam_16_sm(x16)
        up16 = F.interpolate(self.fam_16_up(x16), size=feat8.shape[2:], mode='bilinear', align_corners=True)
        
        x8 = up16 + feat8
        sm8 = self.fam_8_sm(x8)
        up8 = F.interpolate(self.fam_8_up(x8), size=feat4.shape[2:], mode='bilinear', align_corners=True)
        
        sm4 = self.fam_4(up8 + feat4)
        return sm4, sm8, sm16, sm32

@HEADS.register_module()
class FFNetHead(BaseDecodeHead):
    def __init__(self, head_type='A', mid_channels=512, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        # 1. AdapterConv: 将 Backbone 输出的通道压缩到指定的 base_chans
        if head_type == 'A':
            base_chans = [64, 128, 256, 512]
            out_chans = [128, 128, 128, 128]
        else: # B 类型
            base_chans = [64, 128, 128, 256]
            out_chans = [96, 96, 64, 32]

        self.adapter = nn.ModuleList([
            ConvModule(self.in_channels[i], base_chans[i], 1, padding=0, conv_cfg=None, norm_cfg=dict(type='BN'))
            for i in range(len(self.in_channels))
        ])
        
        # 2. UpBranch 特征融合
        self.up_branch = UpBranch(base_chans, out_chans)
        
        # 3. 最后的分类头
        combined_channels = sum(out_chans)
        self.conv_seg = nn.Sequential(
            ConvModule(combined_channels, mid_channels, 1, padding=0, conv_cfg=None, norm_cfg=dict(type='BN')),
            nn.Conv2d(mid_channels, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # 接收 Backbone 输出的 4 个阶段特征
        x = self._transform_inputs(inputs)
        
        # Step 1: 通道适配
        x = [self.adapter[i](x[i]) for i in range(len(x))]
        
        # Step 2: 融合
        x = self.up_branch(x)
        
        # Step 3: 多尺度拼接 (UpsampleCat)
        target_size = x[0].shape[2:]
        out = torch.cat([
            x[0],
            F.interpolate(x[1], size=target_size, mode='bilinear', align_corners=True),
            F.interpolate(x[2], size=target_size, mode='bilinear', align_corners=True),
            F.interpolate(x[3], size=target_size, mode='bilinear', align_corners=True),
        ], dim=1)
        
        # Step 4: 预测
        return self.conv_seg(out)