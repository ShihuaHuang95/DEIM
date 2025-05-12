'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/pdf/2412.16986
'''   

import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
    
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np  
from functools import partial
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
from engine.extre_module.ultralytics_nn.block import C3_Block, C2f_Block, C3k2_Block
  
class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""
     
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):   
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and   
        expansion.  
        """     
        super().__init__()    
        c_ = int(c2 * e)  # hidden channels   
        p = [(2,0,2,0),(0,2,0,2),(0,2,2,0),(2,0,0,2)]  
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        # self.cv1 = nn.ModuleList([nn.Conv2d(c1, c_, k[0], stride=1, padding= p[g], bias=False) for g in range(4)])  
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # y = self.pad[g](x) for g in range(4)     
        return x + self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1))) if self.add else self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = APBottleneck(in_channel, out_channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(YELLOW + '-'*40 + ' C3 ' + '-'*40 + RESET)
    module = C3_Block(in_channel, out_channel, partial(APBottleneck, e=1.0), 2).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(YELLOW + '-'*40 + ' C2f ' + '-'*40 + RESET)   
    module = C2f_Block(in_channel, out_channel, partial(APBottleneck, e=1.0), 2).to(device)     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
     
    print(YELLOW + '-'*40 + ' C3k2 ' + '-'*40 + RESET)   
    module = C3k2_Block(in_channel, out_channel, partial(APBottleneck, e=1.0), 2).to(device)    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
