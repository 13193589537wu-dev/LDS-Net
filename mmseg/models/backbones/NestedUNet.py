# import torch
# import torch.nn as nn
# from mmseg.models.builder import BACKBONES

# def conv_block(in_ch, out_ch, num_convs=2, dilation=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
#     layers = []
#     for i in range(num_convs):
#         layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False))
#         layers.append(nn.BatchNorm2d(out_ch) if norm_cfg['type']=='BN' else nn.Identity())
#         layers.append(nn.ReLU(inplace=True) if act_cfg['type']=='ReLU' else nn.Identity())
#     return nn.Sequential(*layers)

# @BACKBONES.register_module()
# class NestedUNet(nn.Module):
#     """Classic U-Net++ backbone with optional deep supervision and configurable strides."""
#     def __init__(self,
#                  in_channels=3,
#                  base_channels=64,
#                  num_stages=5,
#                  strides=(1,2,2,2,2),
#                  enc_num_convs=(2,2,2,2,2),
#                  dec_num_convs=(2,2,2,2),
#                  downsamples=(True,True,True,True),
#                  enc_dilations=(1,1,1,1,1),
#                  dec_dilations=(1,1,1,1),
#                  deep_supervision=False,
#                  norm_cfg=dict(type='BN'),
#                  act_cfg=dict(type='ReLU')):
#         super().__init__()
#         self.num_stages = num_stages
#         self.deep_supervision = deep_supervision

#         # Encoder blocks
#         self.encoders = nn.ModuleList()
#         in_chs = [in_channels] + [base_channels*2**i for i in range(num_stages-1)]
#         out_chs = [base_channels*2**i for i in range(num_stages)]
#         for i in range(num_stages):
#             self.encoders.append(conv_block(in_chs[i], out_chs[i], num_convs=enc_num_convs[i],
#                                             dilation=enc_dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg))

#         # Downsample layers
#         self.downsamples = nn.ModuleList()
#         for i in range(num_stages-1):
#             if downsamples[i]:
#                 self.downsamples.append(nn.MaxPool2d(strides[i+1]))
#             else:
#                 self.downsamples.append(nn.Identity())

#         # Nested decoder with dense skip connections
#         self.decoders = nn.ModuleDict()
#         for i in range(num_stages-1):
#             for j in range(num_stages-1-i):
#                 in_c = (j+2)*out_chs[i+1] if j>0 else out_chs[i] + out_chs[i+1]
#                 out_c = out_chs[i]
#                 self.decoders[f"x{i}_{j+1}"] = conv_block(in_c, out_c, num_convs=dec_num_convs[i],
#                                                           dilation=dec_dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg)

#     def forward(self, x):
#         x_enc = []
#         out = x
#         # Encoder
#         for i, enc in enumerate(self.encoders):
#             out = enc(out)
#             x_enc.append(out)
#             if i < self.num_stages-1:
#                 out = self.downsamples[i](out)

#         # Decoder
#         x_dec = dict()
#         for i in reversed(range(self.num_stages-1)):
#             for j in range(self.num_stages-1-i):
#                 if j == 0:
#                     x_up = nn.functional.interpolate(x_enc[i+1], scale_factor=2, mode='bilinear', align_corners=True)
#                     x_dec[f"x{i}_{j+1}"] = self.decoders[f"x{i}_{j+1}"](torch.cat([x_enc[i], x_up], dim=1))
#                 else:
#                     ups = [nn.functional.interpolate(x_dec[f"x{i+1}_{k}"], scale_factor=2, mode='bilinear', align_corners=True) for k in range(1,j+1)]
#                     x_cat = torch.cat([x_enc[i]] + ups, dim=1)
#                     x_dec[f"x{i}_{j+1}'] = self.decoders[f"x{i}_{j+1}"](x_cat)

#         # Return final decoder output
#         return x_dec[f"x0_{self.num_stages-1}"] if not self.deep_supervision else list(x_dec.values())
