'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/DRFD.png
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
 
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()     
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)     
        self.act_x = nn.GELU()   
        self.batch_norm_x = nn.BatchNorm2d(out_channels)  
        self.batch_norm_m = nn.BatchNorm2d(out_channels)   
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)   

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]   
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]  
     
        # CutD 
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD 
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)  
        x = self.batch_norm_x(x) 

        # MaxD    
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]     
        m = self.batch_norm_m(m) 

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
    
    module = DRFD(in_channel, out_channel).to(device)
     
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)