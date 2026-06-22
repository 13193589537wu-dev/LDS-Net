import torch
import torch.nn as nn
import torch.nn.functional as F
from .dct_filter import DCT8x8, DCT7x7, DCT3x3

#命名：DCT-based Frequency-Spatial Attention  （D分支）    论文要加
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B,1,H,W]
        x_cat = torch.cat([avg_out, max_out], dim=1)   # [B,2,H,W]
        attn = self.sigmoid(self.conv(x_cat))          # [B,1,H,W]
        return attn


class DCTSA_Single(nn.Module):
    def __init__(self, freq_num, channel, reduction=4, select_method='all'):
        super(DCTSA_Single, self).__init__()
        self.freq_num = freq_num
        self.channel = channel
        self.reduction = reduction
        self.select_method = select_method

        if freq_num == 64:
            self.dct_filter = DCT8x8()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        elif freq_num == 49:
            self.dct_filter = DCT7x7()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        elif freq_num == 9:
            self.dct_filter = DCT3x3()
            self.p = int((self.dct_filter.freq_range - 1) / 2)
        else:
            raise ValueError(f"Unsupported freq_num: {freq_num}")

        if self.select_method == 'all':
            self.dct_c = self.dct_filter.freq_num
        elif 's' in self.select_method:
            self.dct_c = 1
        elif 'top' in self.select_method:
            self.dct_c = int(self.select_method.replace('top', ''))
        else:
            raise ValueError(f"Unsupported select_method: {self.select_method}")

        # 这里只替换掉原来的 FreConv
        self.freq_attention = SpatialAttention(kernel_size=7)

        self.sigmoid = nn.Sigmoid()

        self.out_proj = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def _build_dct_weight(self, x):
        if self.select_method == 'all':
            dct_weight = self.dct_filter.filter
            dct_weight = dct_weight.unsqueeze(1)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        elif 's' in self.select_method:
            filter_id = int(self.select_method.replace('s', ''))
            dct_weight = self.dct_filter.get_filter(filter_id)
            dct_weight = dct_weight.unsqueeze(0).unsqueeze(0)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        elif 'top' in self.select_method:
            filter_id = self.dct_filter.get_topk(self.dct_c)
            dct_weight = self.dct_filter.get_filter(filter_id)
            dct_weight = dct_weight.unsqueeze(1)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        return dct_weight.to(x.device, dtype=x.dtype)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.channel, f'Input channel {c} != expected {self.channel}'

        dct_weight = self._build_dct_weight(x)
        dct_bias = torch.zeros(self.dct_c, device=x.device, dtype=x.dtype)

        dct_feature = F.conv2d(x, dct_weight, dct_bias, stride=1, padding=self.p)

        # SpatialAttention 本身已经带 sigmoid，所以这里不用再套一层 self.sigmoid
        attn_map = self.freq_attention(dct_feature)
# --- 【关键修复：尺寸检查与强行对齐】 ---
        # 检查 attn_map 是否因为卷积 padding 导致尺寸与输入 x 不一致
        if attn_map.shape[-2:] != (h, w):
            attn_map = F.interpolate(attn_map, size=(h, w), mode='bilinear', align_corners=False)
        # ---------------------------------------
        out = x * attn_map + x
        out = self.out_proj(out)

        return out