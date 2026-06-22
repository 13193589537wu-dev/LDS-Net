import torch
from torch import nn
from functools import partial

# from timm.models.layers import trunc_normal_tf_
from timm.models.layers import trunc_normal_ as trunc_normal_tf_
# from timm.models.helpers import named_apply
# from timm.models._manipulate import named_apply
import math

def named_apply(fn, module, name=''):
    for child_name, child in module.named_children():
        full_name = f'{name}.{child_name}' if name else child_name
        named_apply(fn, child, full_name)
    fn(module, name)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class ScaleAttention(nn.Module):
    """
    Scale Attention Module
    为不同 kernel 的 DWConv 输出分配自适应权重
    """
    def __init__(self, channels, num_scales, reduction=4):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Conv2d(channels * num_scales, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_scales, 1, bias=True)
        )

    def forward(self, feats):
        """
        feats: List of [B, C, H, W]
        """
        # GAP for each scale
        pooled = [torch.mean(f, dim=(2, 3), keepdim=True) for f in feats]
        pooled = torch.cat(pooled, dim=1)     # B, C*num_scales, 1, 1

        weights = self.fc(pooled)              # B, num_scales, 1, 1
        weights = torch.softmax(weights, dim=1)

        out = 0
        for i, f in enumerate(feats):
            out = out + f * weights[:, i:i+1]

        return out

class MultiKernelDepthwiseConv(nn.Module):
    """
    多 kernel 深度卷积 + 尺度注意力融合
    """
    def __init__(self, in_channels, kernel_sizes, stride,
                 activation='relu6', dw_parallel=True):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.num_scales = len(kernel_sizes)

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2),
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for k in kernel_sizes
        ])

        # ⭐ 新增尺度注意力
        self.scale_attn = ScaleAttention(in_channels, self.num_scales)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        feats = []

        for dwconv in self.dwconvs:
            out = dwconv(x)
            feats.append(out)

            if not self.dw_parallel:
                x = x + out

        # ⭐ 尺度自适应融合
        out = self.scale_attn(feats)
        return out



class MultiKernelInvertedResidualBlock(nn.Module):
    """多 kernel 逆残差块（MKIR）
    基于MobileNetV2的逆残差结构改进，核心创新：引入多尺度深度卷积
    结构流程：Pointwise Conv（升维）→ 多尺度DWConv → 特征融合 → Channel Shuffle → Pointwise Conv（降维）→ 残差连接
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        # 已经完成多尺度 + 注意力融合
        dout = self.multi_scale_dwconv(pout1)

        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out