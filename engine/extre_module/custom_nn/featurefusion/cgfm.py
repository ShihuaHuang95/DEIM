'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-ContextGuideFusionModule.png    
自研模块：ContextGuideFusionModule
公开讲解视频：https://www.bilibili.com/video/BV1Vx4y1n7hZ/
'''
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch   
import torch.nn as nn
from torch.nn import init

from engine.extre_module.ultralytics_nn.conv import Conv
     
# ContextGuideFusionModule
# 1. 适用任务与解决问题     
# ContextGuideFusionModule 模块专为深度学习框架中的复杂特征融合任务而设计，特别适用于计算机视觉领域，如图像分割、目标检测以及多模态数据整合等任务。该模块旨在解决如何高效融合来自多个源或网络分支的异构特征表示这一核心问题，这些特征通常在通道维度或上下文重点上存在差异。通过协调这些特征，模块有效缓解了信息丢失、特征不对齐以及融合效果欠佳等问题，这些问题往往会削弱模型在需要精确捕捉空间与语义关系的任务中的表现。其自适应的架构确保了互补性上下文线索的稳健整合，使其成为处理特征间复杂相互依赖关系的理想选择。
# 2. 创新点与优势
# ContextGuideFusionModule 引入了一系列开创性的设计，与传统特征融合方法相比，具有显著的创新性和性能优势：    

# 动态通道适配：有别于假设输入维度一致的传统方法，该模块通过自适应卷积层（adjust_conv）对齐不匹配的输入通道。这一创新使其能够无缝融合来自不同网络阶段或模态的特征，极大地拓宽了其在多样化架构中的适用性，且无需繁琐的预处理。
   
# 上下文感知的特征重校准：模块利用挤压-激励（SE）注意力机制，对拼接后的特征进行智能重校准，突出上下文相关的关键信息。这种有针对性的特征增强确保融合过程优先考虑有意义的模式，从而提升融合特征的判别能力。     

# 双向特征增强：模块的一个独特创新在于其跨分支引导机制，即通过互补分支重校准的特征对每个输入特征图进行加权。这种双向交互促进了上下文信息的协同交换，能够捕捉到单向或简单加性融合策略难以发现的复杂依赖关系。
     
# 灵活的输出映射：模块通过条件性的 1x1 卷积（conv1x1）实现输出维度的灵活适配，确保与下游层的兼容性，同时在维持融合表示完整性的前提下提升计算效率。    

# 这些创新带来的优势包括更强的泛化能力，使模块能够稳健处理多样的输入配置；更高的性能表现，通过生成信息丰富的融合特征图显著提升复杂视觉任务的模型精度。此外，其模块化设计便于集成到现有架构中，提供了一种兼顾计算效率与表达能力的可扩展解决方案。

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):   
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),    
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )  
     
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)   
            elif isinstance(m, nn.BatchNorm2d):   
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)   
                if m.bias is not None:
                    init.constant_(m.bias, 0)
  
    def forward(self, x):
        b, c, _, _ = x.size()     
        y = self.avg_pool(x).view(b, c)     
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ContextGuideFusionModule(nn.Module):     
    def __init__(self, inc, ouc) -> None:     
        super().__init__()
    
        self.adjust_conv = nn.Identity() 
        if inc[0] != inc[1]:     
            self.adjust_conv = Conv(inc[0], inc[1], k=1)    
        
        self.se = SEAttention(inc[1] * 2)  
  
        if (inc[1] * 2) != ouc: 
            self.conv1x1 = Conv(inc[1] * 2, ouc)     
        else: 
            self.conv1x1 = nn.Identity() 
    
    def forward(self, x): 
        x0, x1 = x
        x0 = self.adjust_conv(x0)
   
        x_concat = torch.cat([x0, x1], dim=1) # n c h w
        x_concat = self.se(x_concat) 
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1) 
        x0_weight = x0 * x0_weight     
        x1_weight = x1 * x1_weight     
        return self.conv1x1(torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1))
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32  
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
     
    module = ContextGuideFusionModule([channel_1, channel_2], ouc_channel).to(device)  

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],  
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)