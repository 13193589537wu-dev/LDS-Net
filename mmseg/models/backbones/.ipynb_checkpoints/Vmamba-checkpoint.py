import torch
import torch.nn as nn
from mmcv.cnn import  build_norm_layer
from ..builder import BACKBONES

# -----------------------------
# ViMamba Block (官方风格)
# -----------------------------
class ViMambaBlock(nn.Module):
    """Official-style ViMamba block with depthwise conv, pointwise conv, GELU and residual"""
    def __init__(self, dim, drop_path=0.0, norm_cfg=dict(type='BN')):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        # Pointwise MLP
        self.pwconv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-6*torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(self.gamma.view(1,-1,1,1)*x)
        return x

# -----------------------------
# Stem
# -----------------------------
class Stem(nn.Module):
    """Stem layer with conv + norm + activation"""
    def __init__(self, in_chans=3, out_chans=64, norm_cfg=dict(type='BN')):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=7, stride=4, padding=3, bias=False)
        self.norm = build_norm_layer(norm_cfg, out_chans)[1]
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# -----------------------------
# Stage
# -----------------------------
def make_stage(dim, depth, drop_path=0.0, norm_cfg=dict(type='BN')):
    layers = []
    for _ in range(depth):
        layers.append(ViMambaBlock(dim, drop_path=drop_path, norm_cfg=norm_cfg))
    return nn.Sequential(*layers)

# -----------------------------
# ViMamba Backbone
# -----------------------------
@BACKBONES.register_module()
class ViMamba(nn.Module):
    """Official-style ViMamba backbone for segmentation"""
    arch_settings = {
        's12': [64, 128, 320, 512],
        's24': [64, 128, 320, 512],
        's36': [64, 192, 384, 768]
    }

    def __init__(self, arch='s12', in_channels=3, depths=[3,4,6,3],
                 drop_path_rate=0.0, out_indices=(0,1,2,3),
                 norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.depths = depths
        self.dims = self.arch_settings[arch]

        # Stem
        self.stem = Stem(in_chans=in_channels, out_chans=self.dims[0], norm_cfg=norm_cfg)

        # Drop path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stages
        self.stages = nn.ModuleList()
        cur = 0
        for i, dim in enumerate(self.dims):
            stage_depth = depths[i]
            stage_dpr = dpr[cur:cur+stage_depth]
            stage_blocks = nn.Sequential(*[
                ViMambaBlock(dim, drop_path=stage_dpr[j], norm_cfg=norm_cfg) 
                for j in range(stage_depth)
            ])
            self.stages.append(stage_blocks)
            cur += stage_depth
            # Downsample except last stage
            if i < len(self.dims)-1:
                self.stages.append(nn.Conv2d(dim, self.dims[i+1], kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        outs = []
        x = self.stem(x)
        stage_id = 0
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 每两个 module 对应一个 stage（block + downsample）
            if i%2 == 0 and stage_id in self.out_indices:
                outs.append(x)
                stage_id += 1
        return outs

    def init_weights(self, pretrained=None):
        """兼容 MMseg init"""
        if pretrained:
            # 如果有预训练权重可以加载
            checkpoint = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)
class DropPath(nn.Module):
    """DropPath (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# import torch
# import torch.nn as nn
# from ..builder import BACKBONES

# # -----------------------------
# # ViMamba Block
# # -----------------------------
# class ViMambaBlock(nn.Module):
#     """A basic ViMamba block with depthwise conv + pointwise conv"""
#     def __init__(self, dim, norm_cfg=dict(type='BN')):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
#         self.norm = nn.BatchNorm2d(dim) if norm_cfg['type'] == 'BN' else nn.Identity()
#         self.pwconv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
#         self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

#     def forward(self, x):
#         shortcut = x
#         x = self.dwconv(x)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         x = shortcut + self.gamma.view(1,-1,1,1)*x
#         return x

# # -----------------------------
# # Stem
# # -----------------------------
# class Stem(nn.Module):
#     """Initial convolution stem"""
#     def __init__(self, in_chans=3, out_chans=64, norm_cfg=dict(type='BN')):
#         super().__init__()
#         self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=7, stride=4, padding=3, bias=False)
#         self.norm = nn.BatchNorm2d(out_chans) if norm_cfg['type'] == 'BN' else nn.Identity()
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x

# # -----------------------------
# # ViMamba Backbone
# # -----------------------------
# @BACKBONES.register_module()
# class ViMamba(nn.Module):
#     """Classic ViMamba backbone for semantic segmentation"""
#     arch_settings = {
#         's12': [64, 128, 320, 512],
#         's24': [64, 128, 320, 512],
#         's36': [64, 192, 384, 768]
#     }

#     def __init__(self, arch='s12', in_channels=3, depths=[3,4,6,3],
#                  out_indices=(0,1,2,3), norm_cfg=dict(type='BN'), init_cfg=None):
#         super().__init__()
#         self.init_cfg = init_cfg
#         self.out_indices = out_indices
#         self.norm_cfg = norm_cfg
#         self.depths = depths
#         self.dims = self.arch_settings[arch]

#         # Stem
#         self.stem = Stem(in_chans=in_channels, out_chans=self.dims[0], norm_cfg=norm_cfg)

#         # Stages
#         self.stages = nn.ModuleList()
#         for i, dim in enumerate(self.dims):
#             num_blocks = depths[i] if i < len(depths) else 3
#             blocks = []

#             # 如果不是第一个 stage，需要升维
#             if i != 0:
#                 blocks.append(nn.Conv2d(self.dims[i-1], dim, kernel_size=1))

#             # 添加 ViMambaBlock
#             for _ in range(num_blocks):
#                 blocks.append(ViMambaBlock(dim, norm_cfg=norm_cfg))
#             self.stages.append(nn.Sequential(*blocks))

#     def forward(self, x):
#         outs = []
#         x = self.stem(x)
#         for i, stage in enumerate(self.stages):
#             x = stage(x)
#             if i in self.out_indices:
#                 outs.append(x)
#         return outs

#     def init_weights(self, pretrained=None):
#         """兼容 MMseg EncoderDecoder 初始化"""
#         pass
