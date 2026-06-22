import torch
from ..builder import HEADS
from .decode_head import BaseDecodeHead

__all__ = ['EmptyDecodeHead']

@HEADS.register_module()
class EmptyDecodeHead(BaseDecodeHead):
    """
    一个空的解码头适配器。因为 UNet 架构的主干通常已经包含了完整的解码路径，
    并且输出了最终维度的预测结果。
    该类直接透传特征，仅为了让 MMSegmentation 内部框架能正常将特征接入损失函数计算流水线。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # MMSeg 的 BaseDecodeHead 会自动创建一个自带权重的 self.conv_seg，
        # 为了防止在 PyTorch DDP(分布式训练) 环境下报错 “有未使用的参数”，我们必须删除它。
        if hasattr(self, 'conv_seg'):
            del self.conv_seg
    def init_weights(self):
        """重写基类的初始化，因为我们没有 conv_seg，不需要初始化它"""
        pass

    def forward(self, inputs):
        # inputs 接收的是 backbone 返回的列表，MK_UNet 的输出是 [p4]
        # p4 已经是 shape=(B, num_classes, H, W) 的最终 logit。
        return inputs[0]