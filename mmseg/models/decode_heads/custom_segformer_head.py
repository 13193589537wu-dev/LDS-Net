import torch
import torch.nn as nn
from mmseg.models.decode_heads.segformer_head import SegFormerHead
# from mmseg.utils import ConfigType, SampleList
from typing import List, Optional, Union, Dict
import torch.nn.functional as F
from ..builder import HEADS

@HEADS.register_module()
class CustomSegFormerHead(SegFormerHead):
    """SegFormerHead with optional threshold for binary segmentation."""

    def __init__(self,
                 threshold: Optional[float] = None,
                 use_sigmoid: bool = False,
                 **kwargs):
        super(CustomSegFormerHead, self).__init__(**kwargs)
        self.threshold = threshold
        self.use_sigmoid = use_sigmoid

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing."""
        seg_logit = self.forward(inputs)

        if self.use_sigmoid:
            seg_prob = torch.sigmoid(seg_logit)
            if self.threshold is not None:
                seg_pred = (seg_prob > self.threshold).long()
            else:
                seg_pred = (seg_prob > 0.5).long()
        else:
            seg_pred = seg_logit.argmax(dim=1)

        return seg_pred