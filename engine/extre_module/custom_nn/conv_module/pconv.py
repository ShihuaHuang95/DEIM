'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2023-partial convolution.png    
论文链接：https://arxiv.org/pdf/2303.03667
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore') 
from calflops import calculate_flops  

   
import torch
import torch.nn as nn     
from engine.extre_module.ultralytics_nn.conv import Conv

class Partial_Conv(nn.Module): 
    def __init__(self, inc, ouc, n_div=4, forward='split_cat'):
        super().__init__()  
        self.dim_conv3 = inc // n_div 
        self.dim_untouched = inc - self.dim_conv3  
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  
     
        if inc != ouc: 
            self.conv1x1 = Conv(inc, ouc, k=1) # 用作调整通道数
        else:
            self.conv1x1 = nn.Identity()

        if forward == 'slicing':  
            self.forward = self.forward_slicing     
        elif forward == 'split_cat':     
            self.forward = self.forward_split_cat
        else:   
            raise NotImplementedError

    def forward_slicing(self, x):  
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later 
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])   
        return self.conv1x1(x)
    
    def forward_split_cat(self, x):
        # for training/inference     
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)  
        return self.conv1x1(x)     

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  

    module = Partial_Conv(in_channel, out_channel, n_div=4).to(device)
  
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)