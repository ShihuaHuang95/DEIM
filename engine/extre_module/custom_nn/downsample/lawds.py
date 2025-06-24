'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-Light Adaptive-weight downsampling.png   
自研模块：Light Adaptive-weight downsampling
'''   
     
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops 

import torch 
import torch.nn as nn
from einops import rearrange
from engine.extre_module.ultralytics_nn.conv import Conv, autopad

# LAWDS模块描述   
   
# 1. LAWDS模块适合的任务及解决的问题
     
# 轻量自适应权重下采样（Light Adaptive-Weight Downsampling，简称LAWDS）模块专为高维视觉数据的特征提取与降维任务设计，特别适用于需要高效空间分辨率压缩的计算机视觉场景，如图像分类、目标检测和语义分割等。该模块通过引入自适应权重机制，解决了传统下采样方法（如最大池化或卷积下采样）在信息保留与计算效率之间难以平衡的问题，尤其在处理复杂纹理或高频细节时，能够显著减少信息丢失。
     
# LAWDS模块的核心目标是优化下采样过程中的特征选择，使其在降低空间分辨率的同时，动态保留对任务至关重要的语义信息。这对于轻量化模型设计尤为重要，能够在边缘设备或资源受限环境中实现高效推理，同时保持高性能表现。     

# 2. LAWDS模块的创新点与优点 

# 创新点:   

# 自适应权重生成机制：LAWDS模块通过结合全局上下文的注意力机制（基于平均池化和1x1卷积），动态生成空间自适应权重。这种机制突破了传统固定核下采样的局限性，能够根据输入特征的语义内容自适应地调整下采样策略，从而在不同场景下实现更优的特征保留。   
  
# 分组卷积与通道重组：模块采用分组卷积（group convolution）结合通道重组（rearrange）操作，在扩展通道维度的同时降低计算复杂度。这种设计不仅增强了特征表达能力，还通过高效的通道交互保留了跨通道的语义关联。
    
# 多尺度信息融合：通过对下采样特征进行多尺度（s1×s2）重组并施加软最大化（softmax）权重，LAWDS能够在空间维度上实现细粒度的信息加权融合。这种方法在理论上等价于一种局部自注意力机制，但计算开销显著降低，具有更高的工程实用性。
  
# 优点:    

# 高效性与轻量化：LAWDS在保持高性能的同时，通过分组卷积和高效注意力机制大幅减少了参数量和计算量，使其非常适合资源受限的部署场景，如移动端或嵌入式设备。  

# 鲁棒性与通用性：自适应权重机制赋予了模块强大的泛化能力，使其在多样化的视觉任务和数据分布中均能表现出色，尤其是在处理具有高动态范围或复杂背景的图像时。
   
# 综上所述，LAWDS模块通过创新的自适应权重生成与高效特征重组机制，为计算机视觉任务提供了一种兼具高效性、鲁棒性和通用性的下采样解决方案，为轻量化模型设计和边缘计算领域开辟了新的可能性。
     
class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling     
    def __init__(self, in_ch, out_ch, group=16) -> None:
        super().__init__()
     
        self.softmax = nn.Softmax(dim=-1)   
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, in_ch, k=1)  
        )
        
        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=(in_ch // group))     
        self.conv1x1 = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()   
    
    def forward(self, x):     
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)    
     
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)    
        x = torch.sum(x * att, dim=-1)
        return self.conv1x1(x)     
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    module = LAWDS(in_channel, out_channel, group=16).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True)   
    print(RESET)