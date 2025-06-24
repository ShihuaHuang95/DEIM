'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/pdf/2207.02696 
''' 
    
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  

import warnings 
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 
     
import torch    
import torch.nn as nn
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
 
class V7DownSampling(nn.Module):     
    def __init__(self, inc, ouc) -> None:   
        super(V7DownSampling, self).__init__()
 
        ouc = ouc // 2    
        self.maxpool = nn.Sequential(  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            Conv(inc, ouc, k=1)
        )
        self.conv = nn.Sequential(     
            Conv(inc, ouc, k=1),
            Conv(ouc, ouc, k=3, s=2),
        )
    
    def forward(self, x):
        return torch.cat([self.maxpool(x), self.conv(x)], dim=1)  
   
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
   
    module = V7DownSampling(in_channel, out_channel).to(device)    

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
