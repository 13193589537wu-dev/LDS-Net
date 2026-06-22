import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import BACKBONES
import torch.utils.checkpoint as cp


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/ps, W/ps]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (H, W)


class MLP(nn.Module):
    """MLP block"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with Checkpointing"""
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        def attn_fn(norm_x):
            return self.attn(norm_x)
        def mlp_fn(norm_x):
            return self.mlp(norm_x)
        
        x = x + cp.checkpoint(attn_fn, self.norm1(x))  # 移除 use_reentrant
        x = x + cp.checkpoint(mlp_fn, self.norm2(x))   # 移除 use_reentrant
        return x


@BACKBONES.register_module()
class WeakMedSAM(BaseModule):
    """Lightweight Stage-wise WeakMedSAM Backbone for MMSegmentation (Optimized for Memory)"""
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 384],  # Reduced channels
                 depths=[2, 2, 4, 2],             # Reduced depths (6->4 in stage 2)
                 num_heads=[2, 4, 8, 8],          # Reduced heads
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.depths = depths
        self.num_stages = len(embed_dims)

        # Patch embeddings (每个 stage 下采样一次)
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.blocks = nn.ModuleList()

        input_channels = in_channels
        patch_size = 4
        for i in range(self.num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size // (2 ** i),
                patch_size=patch_size if i == 0 else 2,  # 第一个 4x4, 后面2x2
                in_channels=input_channels,
                embed_dim=embed_dims[i]
            )
            self.patch_embeds.append(patch_embed)
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dims[i])))

            stage_blocks = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                ) for _ in range(depths[i])
            ])
            self.blocks.append(stage_blocks)

            input_channels = embed_dims[i]

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)])

        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def forward(self, x):
        B, _, H, W = x.shape
        outs = []
        for i in range(self.num_stages):
            # 如果 x 是 [B, N, C]（上一阶段的 token 输出），转换回 [B, C, H, W]
            if i > 0:  # 第一个 stage 直接接收图像 [B, C, H, W]
                prev_H, prev_W = outs[-1].shape[2], outs[-1].shape[3]  # 获取上一阶段的 H, W
                x = x.transpose(1, 2).reshape(B, self.embed_dims[i-1], prev_H, prev_W)  # 整形回 4 维

            x, (H, W) = self.patch_embeds[i](x)

            # 可插值位置编码
            pos_embed = self.pos_embeds[i]
            if x.shape[1] != pos_embed.shape[1]:
                gs = int(pos_embed.shape[1] ** 0.5)
                pos_embed = pos_embed.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H, W),
                    mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, H * W, -1)

            x = x + pos_embed

            for blk in self.blocks[i]:
                x = blk(x)

            x = self.norms[i](x)
            feat = x.transpose(1, 2).reshape(B, -1, H, W)
            outs.append(feat)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        """初始化权重"""
        if pretrained:
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, map_location='cpu')
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.runner import BaseModule
# from ..builder import BACKBONES


# class PatchEmbed(nn.Module):
#     """Image to Patch Embedding"""
#     def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=64):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim,
#                               kernel_size=patch_size, stride=patch_size)
#         self.num_patches = (img_size // patch_size) ** 2
#         self.embed_dim = embed_dim

#     def forward(self, x):
#         x = self.proj(x)  # [B, C, H/ps, W/ps]
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#         return x, (H, W)


# class MLP(nn.Module):
#     """MLP block"""
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class Attention(nn.Module):
#     """Multi-head Self Attention"""
#     def __init__(self, dim, num_heads=8, qkv_bias=True,
#                  attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Block(nn.Module):
#     """Transformer Block"""
#     def __init__(self, dim, num_heads, mlp_ratio=4.,
#                  qkv_bias=True, drop=0., attn_drop=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             attn_drop=attn_drop,
#             proj_drop=drop
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x


# @BACKBONES.register_module()
# class WeakMedSAM(BaseModule):
#     """Stage-wise WeakMedSAM Backbone for MMSegmentation"""
#     def __init__(self,
#                  img_size=224,
#                  in_channels=3,
#                  embed_dims=[128, 256, 512, 768],
#                  depths=[2, 2, 6, 2],
#                  num_heads=[4, 8, 16, 16],
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  init_cfg=None,
#                  pretrained=None):
#         super().__init__(init_cfg)

#         self.embed_dims = embed_dims
#         self.depths = depths
#         self.num_stages = len(embed_dims)

#         # Patch embeddings (每个 stage 下采样一次)
#         self.patch_embeds = nn.ModuleList()
#         self.pos_embeds = nn.ParameterList()
#         self.blocks = nn.ModuleList()

#         input_channels = in_channels
#         patch_size = 4
#         for i in range(self.num_stages):
#             patch_embed = PatchEmbed(
#                 img_size=img_size // (2 ** i),
#                 patch_size=patch_size if i == 0 else 2,  # 第一个 4x4, 后面2x2
#                 in_channels=input_channels,
#                 embed_dim=embed_dims[i]
#             )
#             self.patch_embeds.append(patch_embed)
#             self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_embed.num_patches, embed_dims[i])))

#             stage_blocks = nn.ModuleList([
#                 Block(
#                     dim=embed_dims[i],
#                     num_heads=num_heads[i],
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     drop=drop_rate,
#                     attn_drop=attn_drop_rate
#                 ) for _ in range(depths[i])
#             ])
#             self.blocks.append(stage_blocks)

#             input_channels = embed_dims[i]

#         self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)])

#         if pretrained:
#             self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

#     def forward(self, x):
#         B, _, H, W = x.shape
#         outs = []
#         for i in range(self.num_stages):
#             x, (H, W) = self.patch_embeds[i](x)

#             # 可插值位置编码
#             pos_embed = self.pos_embeds[i]
#             if x.shape[1] != pos_embed.shape[1]:
#                 gs = int(pos_embed.shape[1] ** 0.5)
#                 pos_embed = pos_embed.reshape(1, gs, gs, -1)
#                 pos_embed = F.interpolate(
#                     pos_embed.permute(0, 3, 1, 2), size=(H, W),
#                     mode='bilinear', align_corners=False
#                 ).permute(0, 2, 3, 1).reshape(1, H * W, -1)

#             x = x + pos_embed

#             for blk in self.blocks[i]:
#                 x = blk(x)

#             x = self.norms[i](x)
#             feat = x.transpose(1, 2).reshape(B, -1, H, W)
#             outs.append(feat)

#         return tuple(outs)

#     def init_weights(self, pretrained=None):
#         """初始化权重"""
#         if pretrained:
#             from mmcv.runner import load_checkpoint
#             load_checkpoint(self, pretrained, strict=False, map_location='cpu')
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.trunc_normal_(m.weight, std=0.02)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.LayerNorm):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
