'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/DWR.png
论文链接：https://arxiv.org/pdf/2212.01173
'''
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops  
    
import torch
import torch.nn as nn 

from engine.extre_module.ultralytics_nn.conv import Conv 

class DWR(nn.Module):
    def __init__(self, inc, ouc) -> None: 
        super().__init__()
     
        if inc != ouc:
            self.conv1x1 = Conv(inc, ouc, 1) 
        else:
            self.conv1x1 = nn.Identity()     

        self.conv_3x3 = Conv(ouc, ouc // 2, 3) 
    
        self.conv_3x3_d1 = Conv(ouc // 2, ouc, 3, d=1)    
        self.conv_3x3_d3 = Conv(ouc // 2, ouc // 2, 3, d=3)     
        self.conv_3x3_d5 = Conv(ouc // 2, ouc // 2, 3, d=5)
   
        self.conv_1x1 = Conv(ouc * 2, ouc, k=1)
        
    def forward(self, x):  
        x = self.conv1x1(x)   
        conv_3x3 = self.conv_3x3(x) 
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)     
        x_out = self.conv_1x1(x_out) + x
        return x_out

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 

    module = DWR(in_channel, out_channel).to(device)
   
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,   
                                     output_precision=4,     
                                     print_detailed=True)     
    print(RESET)