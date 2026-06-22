import torch.nn as nn
from ..builder import BACKBONES
from .resnet import ResNet

@BACKBONES.register_module()
class FFNetResNet(ResNet):
    """
    适配 FFNet 的 ResNet 骨干网络。
    其实就是标准 ResNet 输出 4 个尺度 (1/4, 1/8, 1/16, 1/32)。
    """
    def __init__(self, **kwargs):
        # 强制设置 out_indices 为 4 个阶段
        kwargs['out_indices'] = (0, 1, 2, 3)
        super(FFNetResNet, self).__init__(**kwargs)

    def forward(self, x):
        # 返回 tuple: (feat4, feat8, feat16, feat32)
        return super(FFNetResNet, self).forward(x)