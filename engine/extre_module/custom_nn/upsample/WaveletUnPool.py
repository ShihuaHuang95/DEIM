'''
本文件由BiliBili：魔傀面具整理     
论文链接：https://openreview.net/pdf?id=rkhlb8lCZ    
'''

import warnings 
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import torch  
import torch.nn as nn 
import torch.nn.functional as F   
import numpy as np

class WaveletUnPool(nn.Module):    
    def __init__(self): 
        """ 
        小波反池化 (Wavelet Unpooling) 层，使用 Haar 小波基进行上采样。     
        该层的作用是将输入特征图进行 2x2 上采样，通过小波反变换重构特征。
        """
        super(WaveletUnPool, self).__init__()   
  
        # 定义 Haar 小波的反变换滤波器（低频 LL、高频 LH、HL、HH 分量）   
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])  # 低频成分
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])  # 垂直高频分量
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])  # 水平高频分量
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])  # 对角高频分量
 
        # 组合所有滤波器，并沿第 0 维堆叠 (输出通道数维度)
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
        前向传播函数，执行小波反变换操作。     
        :param x: 输入特征图，形状为 (B, C, H, W)，其中 C 是通道数。    
        :return: 上采样后的特征图，形状为 (B, C/4, 2H, 2W)。
        """ 
   
        # 计算通道数 C，需要保证输入通道数是 4 的倍数，因为每 4 个通道组成一个小波分量    
        C = torch.floor_divide(x.shape[1], 4)  # 计算每个组的通道数
  
        # 复制滤波器，使其适用于所有通道，并扩展到完整的通道数
        filters = torch.cat([self.weight, ] * C, dim=0)   
     
        # 进行反卷积 (转置卷积) 操作，相当于小波反变换
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        
        return y   
 
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)   
    
    module = WaveletUnPool().to(device)     

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)     
    print(RESET)    
