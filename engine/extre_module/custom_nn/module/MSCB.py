'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2024-MSCB.png  
论文链接：https://arxiv.org/abs/2405.06880
'''    
    
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
    
import warnings     
warnings.filterwarnings('ignore')  
from calflops import calculate_flops     

import torch, math 
import torch.nn as nn    
    
from engine.extre_module.ultralytics_nn.conv import Conv  

# MSCB模块     
# 1. MSCB模块适合的任务及解决的问题    
# 多尺度卷积模块（MSCB, Multi-Scale Convolution Block）是一种创新性的深度学习架构，特别适用于需要高效捕获多尺度特征的计算机视觉任务。该模块在图像分类、目标检测、语义分割以及场景理解等任务中表现出色，尤其是在处理复杂场景或具有多样化尺度特征的图像数据时，能够显著提升模型性能。
# MSCB模块通过结合点态卷积（pointwise convolution）和多尺度深度可分离卷积（depthwise separable convolution），解决了传统卷积神经网络在特征提取中的两个关键问题：尺度不变性不足和计算复杂度高。传统卷积操作通常采用单一尺度的卷积核，难以同时捕捉图像中不同尺度的语义信息，而MSCB通过并行或串行的多尺度卷积核设计，能够灵活适应目标对象的尺度变化，从而增强模型对复杂场景的鲁棒性。此外，MSCB模块通过深度可分离卷积显著降低了计算量和参数量，使其特别适合在资源受限的设备（如移动端或嵌入式系统）上部署高效的视觉模型。     
# 在实际应用中，MSCB模块尤其适用于以下场景： 

# 高分辨率图像处理：如医学影像分析中，MSCB能够捕获从细微病灶到整体器官结构的多样化特征。
# 实时视觉任务：如自动驾驶中的目标检测，MSCB能够在保持高精度的同时降低推理延迟。
# 轻量化网络设计：在边缘设备上运行的模型中，MSCB通过高效的多尺度特征提取实现性能与效率的平衡。 
    
# 2. MSCB模块的创新点与优点     
# MSCB模块的创新性体现在其独特的多尺度特征融合机制、高效的计算设计以及灵活的架构适配能力。以下是其核心创新点与优点的详细分析： 
# 创新点   
   
# 多尺度特征自适应融合MSCB通过引入多尺度深度可分离卷积（MSDC），在单一模块内并行或串行地处理多种卷积核尺度（如1x1、3x3、5x5），从而捕获从局部细节到全局语义的多层次特征。与传统的多尺度方法（如Inception模块）不同，MSCB通过深度可分离卷积大幅降低了计算复杂度，同时通过通道混洗（channel shuffle）操作优化了跨尺度特征的交互与融合，提升了特征表达能力。   

# 动态并行/串行模式MSCB模块支持并行或串行的深度卷积模式（通过dw_parallel参数控制），这种灵活性使其能够根据任务需求动态调整特征提取策略。并行模式适合需要同时提取多种尺度特征的场景，而串行模式则通过逐层累加特征增强了模型的深度表达能力。这种设计为网络架构的定制化提供了新的可能性。
 
# 通道混洗优化特征交互MSCB在多尺度特征融合后引入了通道混洗机制，通过分组重排通道的方式增强了不同尺度特征之间的信息流动。这种操作不仅降低了通道间的冗余性，还显著提升了模型对复杂模式的学习能力，尤其是在处理高维特征图时表现突出。     

# 高效的残差连接与扩展因子MSCB通过引入扩展因子（expansion factor）在点态卷积中动态调整通道数，从而在保持轻量化的同时增强了特征表达能力。此外，当步幅为1时，MSCB支持残差连接（skip connection），通过身份映射保留原始输入信息，进一步缓解了深层网络中的梯度消失问题，同时提升了训练稳定性。
   
# 优点     
   
# 高效性与轻量化MSCB采用深度可分离卷积替代传统卷积操作，显著降低了参数量和计算复杂度。例如，与标准3x3卷积相比，深度可分离卷积的计算量可减少至原来的1/9。这种高效性使得MSCB非常适合资源受限场景，同时在高性能硬件上也能实现更快的推理速度。   
 
# 多尺度特征提取的鲁棒性通过多尺度卷积核的协同工作，MSCB能够有效捕获图像中不同尺度和语义层次的特征。这种能力在处理具有尺度变化的目标（如远近不同的物体）或复杂背景的场景时尤为重要，显著提升了模型的泛化能力。     

# 模块化与通用性MSCB作为一个高度模块化的组件，可以无缝集成到现有的卷积神经网络架构中（如ResNet、MobileNet等），无需对整体网络结构进行大幅修改。其灵活的参数配置（如卷积核大小、步幅、扩展因子等）使其能够适配多种任务需求。
  
# 支持边缘部署由于其低计算复杂度和高效的特征提取能力，MSCB特别适合在边缘设备上部署轻量化模型。例如，在移动端的目标检测任务中，MSCB能够在保持高精度的同时显著降低功耗和延迟。   

# 总结   
# MSCB模块通过创新性的多尺度特征提取、高效的深度可分离卷积以及灵活的架构设计，为计算机视觉任务提供了一种兼具高性能和低复杂度的解决方案。其在尺度不变性、计算效率和模型鲁棒性方面的突破，使其在学术研究和工业应用中均具有广阔的前景。无论是用于高精度的图像分类，还是资源受限的边缘计算，MSCB都能为深度学习模型注入新的活力。   


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True): 
        super(MSDC, self).__init__()
  
        self.in_channels = in_channels     
        self.kernel_sizes = kernel_sizes 
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)   
            )
            for kernel_size in self.kernel_sizes  
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []  
        for dwconv in self.dwconvs:   
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:    
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs    

class MSCB(nn.Module):
    """     
    Multi-scale convolution block (MSCB) 
    """   
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super(MSCB, self).__init__() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes   
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel     
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        # check stride value    
        assert self.stride in [1, 2] 
        # Skip connection if stride is 1     
        self.use_skip_connection = True if self.stride == 1 else False
   
        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor) 
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:  
            self.combined_channels = self.ex_channels*1
        else: 
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(   
            # pointwise convolution
            Conv(self.combined_channels, self.out_channels, 1, act=False)  
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
 
    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:     
            dout = 0
            for dwout in msdc_outs:    
                dout = dout + dwout    
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels,self.out_channels)) 
        out = self.pconv2(dout)
        if self.use_skip_connection:   
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out   
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size() 
        channels_per_group = num_channels // groups   
        x = x.view(batchsize, groups, channels_per_group, height, width)    
        x = torch.transpose(x, 1, 2).contiguous()  
        x = x.view(batchsize, -1, height, width)
        return x

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   

    module = MSCB(in_channel, out_channel, kernel_sizes=[1, 3, 5]).to(device)    
  
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,   
                                     output_precision=4, 
                                     print_detailed=True)  
    print(RESET)