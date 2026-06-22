import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class MambaSegHead(BaseDecodeHead):

    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.conv1 = nn.Sequential(

            nn.Conv2d(
                self.in_channels[0],
                self.channels,
                3,
                padding=1),

            nn.BatchNorm2d(self.channels),

            nn.ReLU()
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                self.in_channels[1],
                self.channels,
                3,
                padding=1),

            nn.BatchNorm2d(self.channels),

            nn.ReLU()
        )

        self.fuse = nn.Sequential(

            nn.Conv2d(
                self.channels*2,
                self.channels,
                3,
                padding=1),

            nn.BatchNorm2d(self.channels),

            nn.ReLU()
        )

    def forward(self,inputs):

        inputs = self._transform_inputs(inputs)

        y = inputs[0]

        fuse = inputs[1]

        y = self.conv1(y)

        fuse = self.conv2(fuse)

        fuse = resize(

            fuse,

            size=y.shape[2:],

            mode='bilinear',

            align_corners=self.align_corners
        )

        x = torch.cat([y,fuse],dim=1)

        x = self.fuse(x)

        x = self.cls_seg(x)

        return x