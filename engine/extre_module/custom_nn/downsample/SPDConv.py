''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/SPDConv.png
论文链接：https://arxiv.org/abs/2208.03641     
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 
     
import torch    
import torch.nn as nn
from engine.extre_module.ultralytics_nn.conv import Conv
    
class SPDConv(nn.Module):   
    # Changing the dimension of the Tensor   
    def __init__(self, inc, ouc, dimension=1):  
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)    
  
    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)    
        x = self.conv(x)
        return x  
   
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = SPDConv(in_channel, out_channel).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)  
