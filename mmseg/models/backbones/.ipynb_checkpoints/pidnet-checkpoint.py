import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES
from ..utils.model_utils import BasicBlock, Bottleneck, PagFM, PAPPM, DAPPM
from ..utils.DFSA import DCTSA_Single
from ..utils.CAB import CAB
@BACKBONES.register_module()
class PIDNetBackbone(nn.Module):
    def __init__(self, m=2, n=3, planes=64, ppm_planes=96, **kwargs):
        super(PIDNetBackbone, self).__init__()
        bn_mom = 0.1
        
        # I Branch (Initial & Downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch (Pixel information)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        #-----------------------------------------------------
       # --- 只在 Stage 3、4 之后插入 DFSA (D 支路增强) 加---
        # 根据模型规模判定通道数：Small/Medium 对应 planes，Large 对应 planes * 2
        self.dfsa_stage3 = DCTSA_Single(freq_num=64, channel=32)
        self.dfsa_stage4 = DCTSA_Single(freq_num=64, channel=64)
        #------------------------------------------------------
        
        #------------------------------------------------------------------
        
       # # 在 __init__ 中
        self.cab_pi = CAB(features=128) # 负责 P 和 I 的空间同步
        self.cab_di = CAB(features=128) # 负责 D 和 I 的空间同步
        #------------------------------------------------------------------------
        # D Branch (Detail/Boundary information)
        if m == 2: # Small/Medium
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
        else: # Large
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == (blocks-1))))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)
    # --- 添加下面这个函数以适配 MMSeg ---
    def init_weights(self, pretrained=None):
        """MMSeg 要求的权重初始化入口"""
        if isinstance(pretrained, str):
            from mmcv.runner import load_checkpoint
            from mmseg.utils import get_root_logger
            logger = get_root_logger()
            # 这里的 strict=False 是为了兼容从分类权重加载到分割模型的情况
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # 如果没有预训练权重，执行我们在 __init__ 末尾写的 Kaiming 初始化逻辑
            # 其实在实例化阶段 __init__ 里的逻辑已经运行过了，这里留空即可
            pass
        else:
            raise TypeError('pretrained must be a str or None')
    # ----------------------------------

    def forward(self, x):
        # 记录 I 分支 Layer2 后的尺寸，作为 P 和 D 分支融合的目标尺寸 (1/8)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        
        # 动态获取当前特征图的 H 和 W，不要预先计算 // 8
        target_h, target_w = x.shape[-2:] 
        
        x_p = self.layer3_p(x)
        x_d = self.layer3_d(x)
        
        # Stage 3
        x_i = self.relu(self.layer3(x))
        x_p = self.pag3(x_p, self.compression3(x_i))
        # # 使用动态尺寸
        #   # --- 【核心插入点】：Stage 3 结束
        # # 利用 DCT 提纯 D 支路特征，强化湖泊边界响应
        x_d = self.dfsa_stage3(x_d)
        # -----------------------------------------------
        x_d = x_d + F.interpolate(self.diff3(x_i), size=(target_h, target_w), 
                                 mode='bilinear', align_corners=False)
        temp_p = x_p 
        
        # Stage 4
        x_i = self.relu(self.layer4(x_i))
        x_p = self.layer4_p(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))
        x_p = self.pag4(x_p, self.compression4(x_i))
        #  # --- 加【核心插入点】：Stage 4 结束，Stage 5 开始前 ---
        # # 利用 DCT 提纯 D 支路特征，强化湖泊边界响应
        x_d = self.dfsa_stage4(x_d)   
        # -----------------------------------------------
        x_d = x_d + F.interpolate(self.diff4(x_i), size=(target_h, target_w), 
                                 mode='bilinear', align_corners=False)
        temp_d = x_d 
        
        # Stage 5
        x_p = self.layer5_p(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        
        # SPP 后的输出也要对齐到 1/8 尺寸
        x_i = F.interpolate(self.spp(self.layer5(x_i)), size=(target_h, target_w), 
                           mode='bilinear', align_corners=False)

        
        # # --- 【核心：全对齐逻辑】 ---
        
        # 1. P 分支与 I 分支对齐：纠正细节纹理的位移
        # 注意：这里我们只要对齐后的 P
        x_p_aligned = self.cab_pi(x_p, x_i) 

        # 2. D 分支与 I 分支对齐：纠正边界线的偏移
        # 注意：这里我们只要对齐后的 D
        x_d_aligned = self.cab_di(x_d, x_i) 

        # 3. 现在的 x_p_aligned, x_i, x_d_aligned 三者在空间上是完美重合的
        # 且它们依然保持着各自的功能属性（细节、语义、边界）
        return (x_p_aligned, x_i, x_d_aligned, temp_p, temp_d)
        # return (x_p, x_i, x_d, temp_p, temp_d)