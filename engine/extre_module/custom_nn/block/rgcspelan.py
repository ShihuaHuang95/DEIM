'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-RGCSPELAN.png     
自研模块：RepGhostCSPELAN     
'''   
    
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
  
import torch 
import torch.nn as nn
from engine.extre_module.ultralytics_nn.conv import Conv, RepConv, autopad 
from engine.extre_module.torch_utils import model_fuse_test

# RepGhostCSPELAN模块学术描述   
# 1. RepGhostCSPELAN模块的应用场景与解决的问题
# RepGhostCSPELAN模块是一种高效的多阶段特征提取与聚合架构，专为深度学习中的复杂视觉任务设计。该模块特别适用于需要高精度特征表示的场景，例如目标检测、图像分割和场景理解等计算机视觉任务。针对传统卷积神经网络在特征提取过程中计算复杂度和冗余性较高的问题，RepGhostCSPELAN通过其独特的多路径特征处理机制，优化了特征提取效率，显著降低了计算开销，同时保持甚至提升了模型的表达能力。
# 具体而言，RepGhostCSPELAN能够有效解决以下问题：  

# 特征冗余与计算效率的平衡：通过结合多种卷积操作（如1x1和3x3卷积）与通道分割策略，该模块在保留丰富特征信息的同时，减少了参数量和计算量。
# 多尺度特征融合的不足：模块通过多阶段特征处理和聚合，增强了模型对不同尺度目标的感知能力，特别适合处理具有复杂背景或多尺度目标的视觉任务。     
# 模型轻量化需求：在边缘设备或实时应用场景中，RepGhostCSPELAN能够以较低的计算成本实现高性能特征提取，满足轻量化模型设计的需求。     

# 2. RepGhostCSPELAN模块的创新点与优点     
# RepGhostCSPELAN模块在设计上融入了多项创新性理念，展现出显著的学术价值和工程优势。其创新点和优点主要包括以下几个方面：    
# 创新点  

# 动态通道分割与多路径特征处理RepGhostCSPELAN通过初始1x1卷积实现输入特征的动态通道分割，并结合多路径的3x3卷积处理，构建了灵活的特征提取流程。这种设计不仅增强了特征的多样性，还通过通道缩放机制有效控制了计算复杂度。   
 
# RepConv与Ghost思想的融合模块创新性地将RepConv（可重参数化卷积）与Ghost模块的思想相结合，利用RepConv的结构化稀疏性和Ghost模块的低成本特征生成能力，实现了高效的特征表达。这种融合在推理阶段能够进一步优化模型结构，降低延迟。

# 多阶段特征聚合的层次化设计RepGhostCSPELAN通过多阶段卷积操作和最终的特征拼接，构建了层次化的特征聚合机制。这种设计能够捕捉从低层次纹理到高层次语义的丰富信息，提升了模型对复杂场景的理解能力。 

# 优点    
    
# 高效性与性能的协同优化相较于传统特征提取模块（如CSPNet或SPP），RepGhostCSPELAN在保持高精度的同时，显著降低了计算量和参数量，使其在资源受限场景中具有明显优势。
 
# 模块化与通用性该模块采用模块化设计，易于集成到现有的深度学习框架（如YOLO系列或其他CNN架构）中。其灵活的参数配置（例如通道缩放因子和中间层数）使其能够适配多种任务需求。     
  
# 鲁棒性与适应性通过多路径和多阶段的特征处理，RepGhostCSPELAN展现出对噪声、尺度变化和复杂背景的强大鲁棒性，能够在多样化的视觉任务中稳定表现。 
  
# 综上所述，RepGhostCSPELAN模块以其创新的多路径特征提取、动态通道处理以及高效的计算优化策略，为计算机视觉任务提供了一种兼具高性能和低复杂度的解决方案。其独特的设计理念不仅推动了轻量化神经网络的发展，也为学术界和工业界的模型优化提供了新的思路。


class RGCSPELAN(nn.Module):
    """
    RGCSPELAN: 该模块用于多阶段特征提取和聚合，结合多种卷积操作。    
    
    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。 
        n (int): 额外的中间卷积层数（默认值为1）。   
        scale (float): 中间通道的缩放系数。
        e (float): 隐藏通道的扩展因子。
    """
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RGCSPELAN, self).__init__()  
    
        # 计算中间通道数量
        self.c = int(c2 * e)  # 隐藏通道数    
        self.mid = int(self.c * scale)  # 经过缩放后的中间通道数
        
        # 1x1卷积用于将输入特征拆分为两个部分（后续用于chunk或split） 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
    
        # 最终的1x1卷积层，用于整合所有处理后的特征 
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)     
        
        # 3x3卷积，处理输入特征的第二部分
        self.cv3 = RepConv(self.c, self.mid, 3)
        
        # 一系列额外的3x3卷积层，用于进一步特征提取     
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1)) 
        
        # 1x1卷积，用于进一步处理最后阶段的特征
        self.cv4 = Conv(self.mid, self.mid, 1)
 
    def forward(self, x):  
        """前向传播，使用chunk()方法分割特征图。"""   
        
        # 步骤1: 使用1x1卷积将输入特征拆分成两部分
        y = list(self.cv1(x).chunk(2, 1))   
     
        # 步骤2: 对拆分的第二部分应用3x3卷积  
        y[-1] = self.cv3(y[-1])   
        
        # 步骤3: 依次通过多个3x3卷积进行特征提取   
        y.extend(m(y[-1]) for m in self.m)   
   
        # 步骤4: 使用1x1卷积进一步提取特征
        y.append(self.cv4(y[-1]))   
        
        # 步骤5: 将所有处理后的特征图拼接，并通过最终1x1卷积得到输出
        return self.cv2(torch.cat(y, 1)) 

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = RGCSPELAN(in_channel, out_channel, n=2, scale=0.5, e=0.5).to(device)     
  
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(GREEN + 'test reparameterization.' + RESET)
    module = model_fuse_test(module)   
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET)   

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)