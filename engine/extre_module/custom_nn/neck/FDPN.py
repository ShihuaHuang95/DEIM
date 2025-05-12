'''  
本文件由BiliBili：魔傀面具整理   
自研模块：FocusingDiffusionPyramidNetwork
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
 
import warnings 
warnings.filterwarnings('ignore')     
from calflops import calculate_flops    
  
import copy   
from collections import OrderedDict     
    
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.extre_module.ultralytics_nn.conv import Conv, autopad   
from engine.extre_module.ultralytics_nn.block import C2f    

__all__ = ['FDPN']    
    
class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()    
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)     
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)    

    def forward(self, x):   
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)
   
class FocusFeature(nn.Module):   
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()     
        hidc = int(inc[1] * e)
     
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )     
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()    
        self.conv3 = ADown(inc[2], hidc)
    
   
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, int(hidc / e))
    
    def forward(self, x):
        x1, x2, x3 = x    
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
     
        x = torch.cat([x1, x2, x3], dim=1)   
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)     
  
        x = x + feature 
        return self.conv_1x1(x)

@register(force=True) # 避免因为导入导致的多次注册    
class FDPN(nn.Module):    
    def __init__(self, 
                 in_channels=[512, 1024, 2048],        # 输入特征图的通道数列表，例如来自骨干网络的不同层 
                 feat_strides=[8, 16, 32],             # 输入特征图的步幅列表，表示特征图相对于输入图像的缩放比例
                 hidden_dim=256,                       # 隐藏层维度，所有特征图将被投影到这个维度     
                 nhead=8,                              # Transformer 编码器中多头自注意力的头数
                 dim_feedforward=1024,                 # Transformer 编码器中前馈网络的维度
                 dropout=0.0,                          # Transformer 编码器中的 dropout 概率 
                 enc_act='gelu',                       # Transformer 编码器中的激活函数类型     
                 use_encoder_idx=[2],                  # 指定哪些层使用 Transformer 编码器（索引列表）    
                 num_encoder_layers=1,                 # Transformer 编码器的层数     
                 pe_temperature=10000,                 # 位置编码的温度参数，用于控制频率     
                 fdpn_ks=[3, 5, 7, 9],                 # FDPN中的FocusFeature-kernel_sizes参数
                 depth_mult=1.0,                       # 深度乘数，用于调整网络深度  
                 out_strides=[8, 16, 32],              # 输出特征图的步幅列表 
                 eval_spatial_size=None,               # 评估时的空间尺寸 (H, W)，用于预计算位置编码 
                 ):
        super().__init__()
        from engine.deim.hybrid_encoder import TransformerEncoderLayer, TransformerEncoder # 避免 circular import
    
        # 保存传入的参数为类的成员变量
        self.in_channels = in_channels              # 输入通道数列表
        self.feat_strides = feat_strides            # 输入特征步幅列表
        self.hidden_dim = hidden_dim                # 隐藏层维度
        self.use_encoder_idx = use_encoder_idx      # 使用 Transformer 编码器的层索引 
        self.num_encoder_layers = num_encoder_layers # Transformer 编码器层数
        self.pe_temperature = pe_temperature        # 位置编码温度参数 
        self.eval_spatial_size = eval_spatial_size  # 评估时的空间尺寸
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]  # 输出通道数，统一为 hidden_dim 
        self.out_strides = out_strides              # 输出步幅
     
        assert len(in_channels) == 3 # 仅支持3层特征图的输入

        # 输入投影层：将不同通道数的输入特征图投影到统一的 hidden_dim  
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            # 每个投影层包含 1x1 卷积和批量归一化    
            proj = nn.Sequential(OrderedDict([     
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),  # 1x1 卷积变换通道数    
                ('norm', nn.BatchNorm2d(hidden_dim))                                    # 批量归一化
            ]))   
            self.input_proj.append(proj)
        
        # Transformer 编码器：对指定层进行特征增强     
        # 定义单层 Transformer 编码器
        encoder_layer = TransformerEncoderLayer(   
            hidden_dim,             # 输入维度
            nhead=nhead,            # 注意力头数
            dim_feedforward=dim_feedforward,  # 前馈网络维度 
            dropout=dropout,        # dropout 概率 
            activation=enc_act      # 激活函数
        )  
        # 为每个指定层创建独立的 Transformer 编码器    
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)  # 深拷贝确保独立性
            for _ in range(len(use_encoder_idx))    
        ])
  
        # --------------------------- 第一阶段
        self.FocusFeature_1 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=fdpn_ks)

        self.p4_to_p5_down1 = Conv(hidden_dim, hidden_dim, k=3, s=2)  
        self.p5_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)

        self.p4_to_p3_up1 = nn.Upsample(scale_factor=2) 
        self.p3_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)     

        # --------------------------- 第二阶段    
        self.FocusFeature_2 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=fdpn_ks)
  
        self.p4_to_p5_down2 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)

        if len(out_strides) == 3:     
            self.p4_to_p3_up2 = nn.Upsample(scale_factor=2) 
            self.p3_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)

        # 初始化参数，包括预计算位置编码 
        self._reset_parameters()

    def _reset_parameters(self):
        # 如果指定了评估时的空间尺寸，则预计算位置编码    
        if self.eval_spatial_size: 
            for idx in self.use_encoder_idx:     
                stride = self.feat_strides[idx]  # 当前层的步幅
                # 根据特征图尺寸和步幅计算位置编码   
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,  # 宽度
                    self.eval_spatial_size[0] // stride,  # 高度
                    self.hidden_dim,                      # 嵌入维度     
                    self.pe_temperature                   # 温度参数   
                ) 
                # 将位置编码存储为类的属性
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)    

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        生成 2D sine-cosine 位置编码
        Args: 
            w (int): 特征图宽度
            h (int): 特征图高度
            embed_dim (int): 嵌入维度，必须能被 4 整除
            temperature (float): 温度参数，控制频率
        Returns:
            torch.Tensor: 位置编码张量，形状为 [1, w*h, embed_dim]  
        """
        # 创建宽度和高度的网格     
        grid_w = torch.arange(int(w), dtype=torch.float32)   
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # 生成 2D 网格
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'  
        pos_dim = embed_dim // 4  # 每个方向 (w, h) 的编码维度
        # 计算频率因子
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim     
        omega = 1. / (temperature ** omega)    
   
        # 计算宽度和高度的 sin 和 cos 编码
        out_w = grid_w.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]  
        out_h = grid_h.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]
   
        # 拼接 sin 和 cos 编码，形成最终的位置编码
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :] 

    def forward(self, feats):    
        """
        前向传播函数
        Args:
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致     
        Returns:    
            list[torch.Tensor]: 融合后的多尺度特征图列表
        """
   
        # 检查输入特征图数量是否与预期一致
        assert len(feats) == len(self.in_channels)

        # 输入投影：将所有特征图投影到 hidden_dim 通道
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]  
 
        # Transformer 编码器：对指定层进行特征增强   
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):    
                h, w = proj_feats[enc_ind].shape[2:]  # 获取当前特征图的高度和宽度
                # 将特征图展平并调整维度：[B, C, H, W] -> [B, H*W, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # 根据训练或评估模式选择位置编码     
                if self.training or self.eval_spatial_size is None:
                    # 训练时动态生成位置编码
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)   
                else:
                    # 评估时使用预计算的位置编码  
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)  
     
                # Transformer 编码器处理 
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)  
                # 将输出重塑回特征图形状：[B, H*W, C] -> [B, C, H, W]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        fouce_feature1 = self.FocusFeature_1(proj_feats[::-1]) # 倒序是因为FocusFeature要求从小特征图到大特征图输入
    
        fouce_feature1_to_p5_1 = self.p4_to_p5_down1(fouce_feature1) # fouce_feature1 to p5 
        fouce_feature1_to_p5_2 = self.p5_block1(torch.cat([fouce_feature1_to_p5_1, proj_feats[2]], dim=1))
  
        fouce_feature1_to_p3_1 = self.p4_to_p3_up1(fouce_feature1) # fouce_feature1 to p3 
        fouce_feature1_to_p3_2 = self.p3_block1(torch.cat([fouce_feature1_to_p3_1, proj_feats[0]], dim=1))

        fouce_feature2 = self.FocusFeature_2([fouce_feature1_to_p5_2, fouce_feature1, fouce_feature1_to_p3_2]) 

        fouce_feature2_to_p5 = self.p4_to_p5_down2(fouce_feature2) # fouce_feature2 to p5     
        fouce_feature2_to_p5 = self.p5_block2(torch.cat([fouce_feature2_to_p5, fouce_feature1_to_p5_1, fouce_feature1_to_p5_2], dim=1))

        if len(self.out_strides) == 3:     
            fouce_feature2_to_p3 = self.p4_to_p3_up2(fouce_feature2) # fouce_feature2 to p3 
            fouce_feature2_to_p3 = self.p3_block2(torch.cat([fouce_feature2_to_p3, fouce_feature1_to_p3_1, fouce_feature1_to_p3_2], dim=1))
            return [fouce_feature2_to_p3, fouce_feature2, fouce_feature2_to_p5]  
        else:
            return [fouce_feature2, fouce_feature2_to_p5]
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bs, image_height, image_width = 1, 640, 640
    params = {  
        'in_channels' : [32, 64, 128],
        'feat_strides' : [8, 16, 32],  
        'hidden_dim' : 128,
        'use_encoder_idx' : [2],
        'fdpn_ks' : [3, 5, 7, 9], 
        'depth_mult' : 1.0, 
        'out_strides' : [16, 32],     
        'eval_spatial_size' : [image_height, image_width]  
    }
 
    feats = [torch.randn((bs, params['in_channels'][i], image_height // params['feat_strides'][i], image_width // params['feat_strides'][i])).to(device) for i in range(len(params['in_channels']))]
    module = FDPN(**params).to(device)     
    outputs = module(feats)
   
    input_feats_info = ', '.join([str(i.size()) for i in feats])
    print(GREEN + f'input feature:[{input_feats_info}]' + RESET)    
    output_feats_info = ', '.join([str(i.size()) for i in outputs])
    print(GREEN + f'output feature:[{output_feats_info}]' + RESET)

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats], 
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)