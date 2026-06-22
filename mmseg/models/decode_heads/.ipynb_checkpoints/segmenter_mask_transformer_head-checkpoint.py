import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from ..builder import HEADS


@HEADS.register_module()
class SegmenterMaskTransformerHead(BaseDecodeHead):
    """Segmenter-style Mask Transformer Head (简化版, 适配 mmseg 0.13.0)"""

    def __init__(self,
                 in_channels=768,
                 channels=256,
                 embed_dims=768,
                 num_classes=2,
                 num_heads=8,
                 num_layers=2,
                 dropout_ratio=0.1,
                 align_corners=False,
                 **kwargs):
        super(SegmenterMaskTransformerHead, self).__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            input_transform=None,  # 默认只接收一个 feature map
            **kwargs
        )

        self.embed_dims = embed_dims
        self.num_classes = num_classes

        # 可学习的类别 token
        self.class_emb = nn.Parameter(torch.randn(num_classes, embed_dims))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=int(embed_dims * 4),
            dropout=dropout_ratio,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 初始化
        nn.init.trunc_normal_(self.class_emb, std=0.02)

    def forward(self, inputs, img_metas=None):
        """Forward function."""
        # 将 backbone 输出转换为 (B, N, C)
        x = self._transform_inputs(inputs)  # 这里 inputs 应该是 list[tensor] 或 tensor
        if x.dim() == 4:
            # (B, C, H, W) → (B, N, C)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        B, N, C = x.shape

        # 类别 queries
        class_tokens = self.class_emb.unsqueeze(0).expand(B, -1, -1)

        # Transformer 解码
        feats = x.transpose(0, 1)  # (N, B, C)
        queries = class_tokens.transpose(0, 1)  # (num_classes, B, C)
        hs = self.decoder(queries, feats)  # (num_classes, B, C)
        hs = hs.transpose(0, 1)            # (B, num_classes, C)

        # 还原 feature map
        H = W = int(N ** 0.5)
        feats_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # 计算 mask
        masks = torch.einsum("bqc,bchw->bqhw", hs, feats_2d)

        # resize 到原图大小
        if img_metas is None or img_metas[0] is None:
            # 如果没有 meta 信息，回退到特征图大小
            target_size = (H, W)
        elif 'pad_shape' in img_metas[0] and img_metas[0]['pad_shape'] is not None:
            target_size = img_metas[0]['pad_shape'][:2]
        elif 'img_shape' in img_metas[0] and img_metas[0]['img_shape'] is not None:
            target_size = img_metas[0]['img_shape'][:2]
        else:
            target_size = (H, W)

        masks = resize(
            input=masks,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners
        )

        return masks
