import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils.model_utils import Bag, Light_Bag, segmenthead
from mmseg.ops import resize

@HEADS.register_module()
class PIDHead(BaseDecodeHead):
    def __init__(self, m=2, head_planes=128, channels=128, **kwargs):
        # 显式增加 channels 参数以适配基类要求
        super(PIDHead, self).__init__(
            input_transform='multiple_select', 
            channels=channels, 
            **kwargs)
        
        # 这里的 in_channels[0] 对应的是 backbone 返回的第一个特征图的通道数
        planes = self.in_channels[0] // 2
        
        if m == 2:
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.final_layer = segmenthead(planes * 4, head_planes, self.num_classes)
        
        # Auxiliary heads
        self.seghead_p = segmenthead(planes * 2, head_planes, self.num_classes)
        self.seghead_d = segmenthead(planes * 2, planes, 1)

    def forward(self, inputs):
        """向前传播逻辑，保持返回元组供内部调用"""
        x_p, x_i, x_d, temp_p, temp_d = inputs
        
        # 1. 最终主输出 (Semantic)
        out = self.final_layer(self.dfm(x_p, x_i, x_d))
        
        # 2. 辅助输出 (Training only)
        out_p = self.seghead_p(temp_p)
        out_d = self.seghead_d(temp_d)
        
        return out, out_p, out_d
    # def forward(self, inputs):
    #     """临时简化，只返回一个固定的东西，看能不能过"""
    #     print("进入 PIDHead.forward，inputs 有", len(inputs), "个 tensor")
    #     for i, t in enumerate(inputs):
    #         print(f"  inputs[{i}]: shape={t.shape}, device={t.device}, req_grad={t.requires_grad}")
        
    #     # 临时返回一个假的 logit，形状模仿正常输出
    #     dummy = torch.zeros((inputs[0].shape[0], self.num_classes, inputs[0].shape[2], inputs[0].shape[3]),
    #                         device=inputs[0].device)
    #     return dummy, dummy, dummy   # 让 out, out_p, out_d 都一样

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """重写训练入口，防止基类对元组执行 resize"""
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """推理入口，只返回主分支结果"""
        seg_logit, _, _ = self.forward(inputs)
        return seg_logit

    def get_boundary(self, label):

        if label.dim() == 4:
            label = label.squeeze(1)
    
        with torch.no_grad():
    
            label = label.long()
    
            mask = (label == self.ignore_index)
    
            # 不要用 boolean indexing
            target = torch.where(
                mask,
                torch.zeros_like(label),
                label
            ).float()
    
            target = target.unsqueeze(1)
    
            dilated = F.max_pool2d(
                target,
                3,
                stride=1,
                padding=1
            )
    
            eroded = -F.max_pool2d(
                -target,
                3,
                stride=1,
                padding=1
            )
    
            edge = (dilated - eroded > 0).float()
    
            mask = mask.unsqueeze(1)
    
            edge = torch.where(
                mask,
                torch.zeros_like(edge),
                edge
            )
    
        return edge

    def losses(self, seg_logits, gt_semantic_seg):
        
        
        # 然后继续你的代码，但暂时不要调用 get_boundary
    
        seg_logit, seg_logit_p, edge_logit = seg_logits
        losses = dict()
        target_size = gt_semantic_seg.shape[2:]
        
        # 确保标签是 Long 类型用于 CrossEntropy
        labels = gt_semantic_seg.squeeze(1).long()

        ignore_mask = labels==255
        
        labels = torch.clamp(labels,0,self.num_classes-1)

        # 1. 主损失
        seg_logit = resize(seg_logit, size=target_size, mode='bilinear', align_corners=self.align_corners)
        losses['loss_semantic'] = self.loss_decode[0](seg_logit, labels)
            
        # 2. 辅助损失
        seg_logit_p = resize(seg_logit_p, size=target_size, mode='bilinear', align_corners=self.align_corners)
        losses['loss_aux'] = self.loss_decode[1](seg_logit_p, labels)
            
        #3. 边界损失
        edge_logit = resize(
            edge_logit,
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners
        )
        
        # PIDNet 必须 sigmoid
        edge_logit = torch.sigmoid(edge_logit)
        
        edge_label = self.get_boundary(labels)
        
        edge_label = edge_label.float().detach()
        
        # 防止数值爆炸
        edge_logit = torch.clamp(edge_logit,0,1)
        
        losses['loss_boundary'] = self.loss_decode[2](
            edge_logit,
            edge_label
        )
        for k,v in losses.items():
            if torch.isnan(v):
                print("NaN loss:",k)
                losses[k]=torch.zeros_like(v)
        return losses

    def init_weights(self):
        """初始化逻辑"""
        # PIDNet 通常在内部已完成初始化，这里调用基类即可
        super(PIDHead, self).init_weights()