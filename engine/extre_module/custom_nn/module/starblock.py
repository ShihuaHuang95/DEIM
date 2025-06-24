'''    
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/CVPR2024-StarBlock.png
论文链接：https://arxiv.org/pdf/2403.19967
'''   
    
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
   
import torch     
import torch.nn as nn
from timm.layers import DropPath   
     
from engine.extre_module.ultralytics_nn.conv import Conv   
 
class Star_Block(nn.Module):    
    def __init__(self, inc, ouc, mlp_ratio=3, drop_path=0.):
        super().__init__()   
        self.dwconv = Conv(inc, inc, 7, g=inc, act=False)  
        self.f1 = nn.Conv2d(inc, mlp_ratio * inc, 1)   
        self.f2 = nn.Conv2d(inc, mlp_ratio * inc, 1)    
        self.g = Conv(mlp_ratio * inc, inc, 1, act=False)   
        self.dwconv2 = nn.Conv2d(inc, inc, 7, 1, (7 - 1) // 2, groups=inc) 
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
  
        if inc != ouc:
            self.conv1x1 = Conv(inc, ouc, k=1)    
        else:
            self.conv1x1 = nn.Identity()  

    def forward(self, x):   
        input = x 
        x = self.dwconv(x)    
        x1, x2 = self.f1(x), self.f2(x)     
        x = self.act(x1) * x2  
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return self.conv1x1(x)   
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = Star_Block(in_channel, out_channel, mlp_ratio=3).to(device)     

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)    
    print(RESET)