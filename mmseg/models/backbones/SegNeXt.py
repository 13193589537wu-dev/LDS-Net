import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES

# -----------------------------
# Multi-Scale Convolution Attention (MSCA)
# -----------------------------
class MSCA(nn.Module):
    """Multi-Scale Convolutional Attention (核心模块)"""
    def __init__(self, dim):
        super().__init__()
        # 多尺度 depthwise 卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 7, padding=15, dilation=5, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 7, padding=21, dilation=7, groups=dim)

        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        x = self.conv0(x)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = self.pwconv(x)
        return x * u

# -----------------------------
# MSCAN Block
# -----------------------------
class MSCABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MSCA(dim)
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

        self.norm2 = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
        x = x + self.gamma2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))
        return x

# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=stride, stride=stride)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

# -----------------------------
# SegNeXt (MSCAN backbone)
# -----------------------------
@BACKBONES.register_module()
class MSCAN(nn.Module):
    arch_settings = {
        's': [64, 128, 320, 512],   # small
        'b': [64, 128, 320, 512],   # base
        'l': [64, 128, 320, 512],   # large
    }

    def __init__(self, arch='s', in_chans=3, depths=[3, 3, 9, 3],
                 out_indices=(0, 1, 2, 3), init_cfg=None):
        super().__init__()
        self.out_indices = out_indices
        self.init_cfg = init_cfg
        dims = self.arch_settings[arch]

        # Stem & Patch Embedding
        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_dim = in_chans
        for i, dim in enumerate(dims):
            stride = 4 if i == 0 else 2
            self.patch_embeds.append(PatchEmbed(in_dim, dim, stride))
            blocks = [MSCABlock(dim) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))
            in_dim = dim

    def forward(self, x):
        outs = []
        for i, (patch_embed, stage) in enumerate(zip(self.patch_embeds, self.stages)):
            x = patch_embed(x)
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def init_weights(self, pretrained=None):
        """兼容 MMSeg 初始化方式"""
        if isinstance(pretrained, str):
            print(f"Load pretrained from {pretrained} (请用 mmcv 的 load_checkpoint)")
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
