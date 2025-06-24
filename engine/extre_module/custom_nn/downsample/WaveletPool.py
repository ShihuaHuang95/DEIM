'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/ICLR2018-WaveletPool.png   
论文链接：https://openreview.net/pdf?id=rkhlb8lCZ    
'''   
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')   
from calflops import calculate_flops 

import torch
import torch.nn as nn
import torch.nn.functional as F     
import numpy as np    
    
from engine.extre_module.ultralytics_nn.conv import Conv

class WaveletPool(nn.Module):   
    def __init__(self):
        """     
        小波池化 (Wavelet Pooling) 层，使用 Haar 小波基进行 2x2 下采样。 
        该层的作用是将输入特征图降采样，并将其转换为小波系数（低频 LL 和高频 LH、HL、HH 分量）。  
        """
        super(WaveletPool, self).__init__()
        
        # 定义 Haar 小波的变换滤波器（低频 LL、高频 LH、HL、HH 分量）
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])  # 低频分量     
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])  # 垂直高频分量
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])  # 水平高频分量   
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])  # 对角高频分量    
     
        # 组合所有滤波器，并沿第 0 维度 (输出通道维度) 堆叠    
        filts = np.stack([
            ll[None, ::-1, ::-1],  # 低频分量 (LL)
            lh[None, ::-1, ::-1],  # 垂直高频分量 (LH)
            hl[None, ::-1, ::-1],  # 水平高频分量 (HL)  
            hh[None, ::-1, ::-1]   # 对角高频分量 (HH)
        ], axis=0)
        
        # 将滤波器转换为 PyTorch 张量，并设为不可训练参数   
        self.weight = nn.Parameter(   
            torch.tensor(filts).to(torch.get_default_dtype()),  # 转换为默认数据类型
            requires_grad=False  # 该参数在训练过程中不进行更新
        )

    def forward(self, x): 
        """    
        前向传播函数，执行小波变换池化操作。  
        :param x: 输入特征图，形状为 (B, C, H, W)，其中 C 是通道数。   
        :return: 下采样后的特征图，形状为 (B, 4C, H/2, W/2)，其中 4C 代表 4 个小波分量。
        """  
   
        # 获取输入的通道数 C，每个通道都会被分解为 4 个小波分量    
        C = x.shape[1]  # 输入特征图的通道数   
        
        # 复制滤波器，使其适用于所有通道，并扩展到完整的通道数 
        filters = torch.cat([self.weight, ] * C, dim=0)
        
        # 进行 2D 卷积 (相当于小波变换)，步长 2 进行 2x2 下采样
        y = F.conv2d(x, filters, groups=C, stride=2)   
     
        return y   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = WaveletPool().to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)    
