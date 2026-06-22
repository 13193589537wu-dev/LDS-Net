import torch
import torch.nn as nn
from ..builder import BACKBONES

# -----------------------------
# SegNeXt Block
# -----------------------------
class SegNeXtBlock(nn.Module):
    """Simplified SegNeXt Block"""
    def __init__(self, dim, norm_cfg=dict(type='BN')):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim) if norm_cfg['type'] == 'BN' else nn.Identity()
        self.pwconv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-6*torch.ones(dim), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.gamma.view(1,-1,1,1)*x
        return x

# -----------------------------
# Stem
# -----------------------------
class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, norm_cfg=dict(type='BN')):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=7, stride=4, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_chans) if norm_cfg['type']=='BN' else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# -----------------------------
# SegNeXt Backbone
# -----------------------------
@BACKBONES.register_module()
class SegNeXt(nn.Module):
    arch_settings = {
        's12': [64, 128, 320, 512],
        's24': [64, 128, 320, 512],
        's36': [64, 192, 384, 768]
    }

    def __init__(self, arch='s12', in_channels=3, depths=[3,4,6,3],
                 out_indices=(0,1,2,3), norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.depths = depths
        self.dims = self.arch_settings[arch]

        # Stem
        self.stem = Stem(in_chans=in_channels, out_chans=self.dims[0], norm_cfg=norm_cfg)

        # Stages
        self.stages = nn.ModuleList()
        for i, dim in enumerate(self.dims):
            num_blocks = depths[i] if i < len(depths) else 3
            blocks = []

            # 如果不是第一个 stage，需要升维
            if i != 0:
                blocks.append(nn.Conv2d(self.dims[i-1], dim, kernel_size=1))

            # 添加 SegNeXtBlock
            for _ in range(num_blocks):
                blocks.append(SegNeXtBlock(dim, norm_cfg=norm_cfg))
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def init_weights(self, pretrained=None):
        """兼容 MMseg EncoderDecoder 初始化"""
        pass
