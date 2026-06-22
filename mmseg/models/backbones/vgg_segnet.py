# mmseg/models/backbones/vgg_backbone.py
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import BACKBONES


@BACKBONES.register_module()
class VGGBackbone(BaseModule):
    """
    VGG16-like backbone that returns a tuple of feature maps.
    - 支持 out_indices 参数（tuple of ints），用于选择需要返回的 stage 输出（与 mmseg 习惯一致）。
    - 接受额外 kwargs（如 dilations, etc.）以与 config 兼容。
    """

    def __init__(
        self,
        in_channels=3,
        depth=16,
        with_bn=True,
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        pretrained=None,
        init_cfg=None,
        **kwargs,  # 接受并忽略多余 config 参数，防止 unexpected keyword 错误
    ):
        super(VGGBackbone, self).__init__(init_cfg=init_cfg)
        assert depth in (11, 13, 16, 19)
        self.in_channels = in_channels
        self.depth = depth
        self.with_bn = with_bn
        self.num_stages = num_stages
        # normalize out_indices 为 tuple
        if out_indices is None:
            out_indices = tuple(range(num_stages))
        self.out_indices = tuple(out_indices)

        cfgs = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        layers_cfg = cfgs[depth]

        stages = []
        cur_layers = []
        in_c = in_channels
        for v in layers_cfg:
            if v == 'M':
                stages.append(nn.Sequential(*cur_layers))
                cur_layers = []
            else:
                conv = nn.Conv2d(in_c, v, kernel_size=3, padding=1)
                modules = [conv]
                if with_bn:
                    modules.append(nn.BatchNorm2d(v))
                modules.append(nn.ReLU(inplace=True))
                cur_layers.extend(modules)
                in_c = v
        if len(cur_layers) > 0:
            stages.append(nn.Sequential(*cur_layers))

        # 保证 stages 数量不超过 num_stages
        self.stages = nn.ModuleList(stages[:num_stages])

        # record stage out channels (最后 conv 的 out channels)
        self.stage_out_channels = []
        for s in self.stages:
            last_conv = None
            for m in s.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            self.stage_out_channels.append(last_conv.out_channels if last_conv is not None else in_channels)

        # 校验 out_indices 合法性
        max_idx = len(self.stages) - 1
        for idx in self.out_indices:
            if not (0 <= idx <= max_idx):
                raise ValueError(f'out_indices contains invalid index {idx}, allowed range 0..{max_idx}')

    def forward(self, x: torch.Tensor):
        feats = []
        out = x
        for i, stage in enumerate(self.stages):
            out = stage(out)
            # 通常 VGG 在 stage 结束后会下采样（MaxPool），但这里保留 stage 输出供上层决定是否下采样/使用。
            # 若需要下采样行为，可在 config/backbone 中自定义 stage 末尾包含 MaxPool。
            feats.append(out)

        # 返回 mmseg 期望的 tuple（按照 out_indices 选择）
        selected = tuple(feats[i] for i in self.out_indices)
        return selected


    def init_weights(self, pretrained=None):
        """
        初始化权重。兼容 mmseg/encoder_decoder 中传入 pretrained 参数。
        - pretrained: None 或 str 路径/URI。如果给出字符串，这里暂不做复杂加载（可扩展），
          目前实现为：若为 None，执行 kaiming 初始化；若为字符串，尝试用 torch.load 加载 state_dict（若合适）。
        """
        # 如果用户没有传预训练路径，则使用 kaiming 初始化当前模块的卷积/反卷积层等
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            return

        # 如果传入了字符串，尝试加载（这是一个简单尝试，可能需要你根据权重格式调整）
        if isinstance(pretrained, str):
            try:
                import torch
                state = torch.load(pretrained, map_location='cpu')
                # 如果 state 是 dict 且含 'state_dict'，取出
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                # 试图加载到当前模块
                missing, unexpected = self.load_state_dict(state, strict=False)
                # 可选：打印信息
                try:
                    from mmcv.utils import print_log
                    print_log(f'Loaded pretrained backbone from {pretrained}. missing: {len(missing)}, unexpected: {len(unexpected)}')
                except Exception:
                    print(f'Loaded pretrained backbone from {pretrained}. missing: {len(missing)}, unexpected: {len(unexpected)}')
                return
            except Exception as e:
                # 如果加载失败，回退到默认初始化并告警
                try:
                    from mmcv.utils import print_log
                    print_log(f'Warning: failed to load pretrained weights from {pretrained}: {e}')
                except Exception:
                    print(f'Warning: failed to load pretrained weights from {pretrained}: {e}')
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                return

        # 其它类型直接按默认初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
