'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/SRFD.png
论文链接：https://ieeexplore.ieee.org/document/10142024   
'''    
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 

import torch
import torch.nn as nn
    
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super().__init__()  
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)  
        self.batch_norm = nn.BatchNorm2d(out_channels)
 
    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]   
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x    

class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2 
  
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)    

        # original size to 2x downsampling layer     
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)   
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)   
        self.cut_c = Cut(out_c14, out_c12)   
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)  

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)   
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)     
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels) 
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)  
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)    
  
    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]    

    # original size to 2x downsampling layer    
        c = x                   # c = [B, C/4, H, W]    
        # CutD
        c = self.cut_c(c)       # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]   
        # ConvD
        x = self.conv_1(x)      # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2] 
        x = self.conv_x1(x)     # x = [B, C/2, H/2, W/2]  
        x = self.batch_norm_x1(x)   
        # Concat + conv
        x = torch.cat([x, c], dim=1)    # x = [B, C, H/2, W/2]
        x = self.fusion1(x)     # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]     
   
    # 2x to 4x downsampling layer
        r = x                   # r = [B, C/2, H/2, W/2]     
        x = self.conv_2(x)      # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]    
        m = x                   # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)     # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD    
        m = self.max_m(m)       # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)       # r = [B, C, H/4, W/4]
        # Concat + conv  
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)     # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]  
        return x                # x = [B, C, H/4, W/4]
  
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    module = SRFD(in_channel, out_channel).to(device)
    
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True, 
                                     output_precision=4, 
                                     print_detailed=True)     
    print(RESET) 
