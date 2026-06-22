# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import torch.nn as nn
@HEADS.register_module()
class SimpleHead(BaseDecodeHead):
    def __init__(self, is_dw=False, **kwargs):
        super(SimpleHead, self).__init__(input_transform='multiple_select', **kwargs)

        embedding_dim = self.channels # 对应 config 中的 channels=160

        # 添加：为每个输入分支创建一个 1x1 卷积，统一通道数
        self.convs = nn.ModuleList()
        for in_channel in self.in_channels:
            self.convs.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def agg_res(self, preds):
        # 先统一通道数
        transformed_preds = []
        for i, pred in enumerate(preds):
            transformed_preds.append(self.convs[i](pred))
            
        # 再进行缩放和相加
        outs = transformed_preds[0]
        for pred in transformed_preds[1:]:
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs = outs + pred # 使用 + 避免 inplace 操作导致梯度问题
        return outs

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        return x
        
# @HEADS.register_module()
# class SimpleHead(BaseDecodeHead):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, is_dw=False, **kwargs):
#         super(SimpleHead, self).__init__(input_transform='multiple_select', **kwargs)

#         embedding_dim = self.channels

#         self.linear_fuse = ConvModule(
#             in_channels=embedding_dim,
#             out_channels=embedding_dim,
#             kernel_size=1,
#             stride=1,
#             groups=embedding_dim if is_dw else 1,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg
#         )
    
#     def agg_res(self, preds):
#         outs = preds[0]
#         for pred in preds[1:]:
#             pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
#             outs += pred
#         return outs

#     def forward(self, inputs):
#         xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
#         x = self.agg_res(xx)
#         _c = self.linear_fuse(x)
#         x = self.cls_seg(_c)
#         return x