import torch
import torch.nn as nn
from ..builder import BACKBONES

# --------------------
# 基础模块
# --------------------
class LayerNorm2d(nn.Module):
    """LayerNorm in channel-last style but applied on 2D feature maps"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MLP(nn.Module):
    """MLP used in PoolFormer"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Pooling(nn.Module):
    """PoolFormer Token Mixer"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    """一个标准 PoolFormer Block"""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = Pooling()
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

        # 可学习缩放因子
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # Token mixing
        x = x + self.gamma1.view(1, -1, 1, 1) * self.token_mixer(self.norm1(x))
        # MLP
        x = x + self.gamma2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding (Conv Downsample)"""
    def __init__(self, in_chans, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


# --------------------
# PoolFormer Backbone
# --------------------
@BACKBONES.register_module()
class PoolFormer(nn.Module):
    arch_settings = {
        's12': [2, 2, 6, 2],   # 每个 stage 的 block 数
        's24': [4, 4, 12, 4],
        's36': [6, 6, 18, 6],
        'm36': [6, 6, 18, 6],
    }
    dims = {
        's12': [64, 128, 320, 512],
        's24': [64, 128, 320, 512],
        's36': [64, 128, 320, 768],
        'm36': [96, 192, 384, 768],
    }

    def __init__(self, arch='s12', in_chans=3, out_indices=(0,1,2,3), init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        depths = self.arch_settings[arch]
        dims = self.dims[arch]

        # Stem
        self.patch_embed = PatchEmbed(in_chans, dims[0], patch_size=7, stride=4, padding=3)

        # Stages
        self.stages = nn.ModuleList()
        in_dim = dims[0]
        for i in range(len(depths)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(PoolFormerBlock(dims[i]))
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            if i < len(depths)-1:
                # Patch embedding (downsample)
                downsample = PatchEmbed(dims[i], dims[i+1], patch_size=3, stride=2, padding=1)
                self.stages.append(downsample)

    def forward(self, x):
        outs = []
        x = self.patch_embed(x)
        stage_id = 0
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if isinstance(layer, nn.Sequential):  # block stage
                if stage_id in self.out_indices:
                    outs.append(x)
                stage_id += 1
        return outs

    def init_weights(self, pretrained=None):
        # TODO: 可以接上 mmcv.load_checkpoint(pretrained, self) 来加载官方预训练
        pass
