import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES,build_loss
from .dice_loss import DiceLoss
from .cross_entropy_loss import CrossEntropyLoss

@LOSSES.register_module()
class MultiLoss(nn.Module):
    """MultiLoss 支持组合多种 loss，例如 CrossEntropy + DiceLoss"""
    def __init__(self, losses, **kwargs):
        super(MultiLoss, self).__init__()
        self.losses = nn.ModuleList()
        for loss_cfg in losses:
            loss_cfg = loss_cfg.copy()
            for k, v in kwargs.items():
                if k not in loss_cfg:
                    loss_cfg[k] = v
            self.losses.append(build_loss(loss_cfg))

    def forward(self, pred, target, **kwargs):
        """计算组合 loss"""
        total_loss = 0
        loss_dict = {}
        for loss_module in self.losses:
            loss_name = loss_module.__class__.__name__
            loss_value = loss_module(pred, target, **kwargs)
            # 保证返回 tensor 类型
            if isinstance(loss_value, dict):
                for k, v in loss_value.items():
                    loss_dict[k] = v
                total_loss += sum(loss_value.values())
            else:
                loss_dict[loss_name] = loss_value
                total_loss += loss_value
        # 关键改动：返回 tensor，而不是 dict
        return total_loss  # MMSeg _parse_losses 可以识别