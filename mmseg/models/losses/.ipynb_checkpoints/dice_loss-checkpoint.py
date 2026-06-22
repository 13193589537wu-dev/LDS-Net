import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES  # 若你是 MMSeg < 1.0，请改为 from mmseg.models.losses import LOSSES
from mmcv.runner import BaseModule
@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=False, eps=1e-6, **kwargs):
        """
        Dice Loss 支持多类别或二分类，并兼容 MultiLoss 传入的多余参数
        Args:
            use_sigmoid (bool): 是否对预测使用 sigmoid
            eps (float): 防止除零
            **kwargs: 兼容多余参数（如 loss_weight, ignore_index）
        """
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, pred, target, ignore_index=None, **kwargs):
        """
        pred: [N, C, H, W] 预测
        target: [N, H, W] 标签
        ignore_index: int, 可选，忽略标签值
        """
        if ignore_index is not None:
            mask = target != ignore_index
            target = target * mask.long()

        # one-hot 编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # [N,H,W,C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [N,C,H,W]

        if self.use_sigmoid:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)

        if ignore_index is not None:
            mask = mask.unsqueeze(1).float()  # [N,1,H,W]
            pred = pred * mask
            target_one_hot = target_one_hot * mask

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice
        return loss.mean()
# class DiceLoss(nn.Module):
#     """Dice Loss for semantic segmentation with ignore_index and class weights."""
#     def __init__(self,
#                  use_sigmoid=False,
#                  reduction='mean',
#                  class_weight=None,
#                  loss_weight=1.0,
#                  eps=1e-6,
#                  **kwargs):  # 加上 **kwargs
#         super(DiceLoss, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.eps = eps
#         print(">>> Custom DiceLoss loaded!")
#         if class_weight is not None:
#             self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
#         else:
#             self.class_weight = None

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """
#         pred: [N, C, H, W] logits or probabilities
#         target: [N, H, W] with values in [0, num_classes-1]
#         weight: optional sample-wise weight
#         kwargs: may contain 'ignore_index'
#         """
#         reduction = reduction_override if reduction_override else self.reduction
#         ignore_index = kwargs.get('ignore_index', None)

#         pred = pred.contiguous()
#         target = target.contiguous()

#         # Apply activation
#         if self.use_sigmoid:
#             pred = torch.sigmoid(pred)
#             if pred.shape[1] == 1:
#                 pred = pred.squeeze(1)  # [N,H,W]
#         else:
#             pred = F.softmax(pred, dim=1)

#         num_classes = pred.shape[1]

#         # Handle ignore_index
#         if ignore_index is not None:
#             mask = (target != ignore_index)
#             target = target.clone()
#             target[~mask] = 0  # 临时替换 ignore_index 为 0，后续 loss 时会忽略
#         else:
#             mask = torch.ones_like(target, dtype=torch.bool)

#         # Convert target to one-hot
#         if pred.ndim == 4 and pred.shape[1] > 1:
#             if target.ndim == 3:
#                 target = F.one_hot(target, num_classes=num_classes)  # [N,H,W,C]
#                 target = target.permute(0, 3, 1, 2).float()  # [N,C,H,W]

#         # Apply mask
#         if ignore_index is not None:
#             mask = mask.unsqueeze(1)  # [N,1,H,W]
#             pred = pred * mask
#             target = target * mask

#         # Compute Dice
#         intersection = (pred * target).sum(dim=(2,3))  # [N,C]
#         union = (pred + target).sum(dim=(2,3))  # [N,C]
#         dice_score = (2 * intersection + self.eps) / (union + self.eps)
#         loss = 1 - dice_score  # [N,C]

#         # Apply class weights
#         if self.class_weight is not None:
#             class_weight = self.class_weight.to(pred.device)
#             loss = loss * class_weight  # [N,C] * [C]

#         # Mean across channels
#         loss = loss.mean(dim=1)  # [N]

#         # Apply sample-wise weight
#         if weight is not None:
#             weight = weight.to(loss.device).float()
#             loss = loss * weight

#         # Final reduction
#         if reduction == 'mean':
#             loss = loss.mean() if avg_factor is None else loss.sum() / avg_factor
#         elif reduction == 'sum':
#             loss = loss.sum()

#         return self.loss_weight * loss