import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES # 如果是旧版用 from .builder import LOSSES

@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """PIDNet Boundary Loss 实现"""
    def __init__(self, loss_weight=1.0, loss_name='loss_boundary'):
        super(BoundaryLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                preds,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """
        preds: (B, 1, H, W) 预测的边界图
        target: (B, H, W) 真实的边界标签 (0或1)
        """
        # 确保 target 维度匹配 preds
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 将 target 转为 float 用于计算 BCE
        target = target.float()
        
        # 使用带 Sigmoid 的二进制交叉熵
        loss = F.binary_cross_entropy_with_logits(
            preds, target, reduction='mean')
        
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name