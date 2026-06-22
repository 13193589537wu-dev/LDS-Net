import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import sys, os
from inplace_abn import InPlaceABN, InPlaceABNSync
class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            InPlaceABNSync(features),

            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            InPlaceABNSync(features),

            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            InPlaceABNSync(features),

            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen1[-1].weight.data.zero_()
        self.delta_gen2[-1].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0

        norm = torch.tensor([[[[w / s, h / s]]]], dtype=input.dtype, device=input.device)

        w_list = torch.linspace(-1.0, 1.0, out_h, device=input.device, dtype=input.dtype).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w, device=input.device, dtype=input.dtype).repeat(out_h, 1)

        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0

        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]],
                            dtype=input.dtype, device=input.device)

        w_list = torch.linspace(-1.0, 1.0, out_h, device=input.device, dtype=input.dtype).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w, device=input.device, dtype=input.dtype).repeat(out_h, 1)

        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)

        high_stage = F.interpolate(
            input=high_stage,
            size=(h, w),
            mode='bilinear',
            align_corners=True
        )

        concat = torch.cat((low_stage, high_stage), dim=1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)

        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        out = high_stage + low_stage
        return out