'''   
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2024-EUCB.png  
论文链接：https://arxiv.org/abs/2405.06880
'''  
     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
     
import torch
import torch.nn as nn 
  
from engine.extre_module.ultralytics_nn.conv import Conv

class EUCB(nn.Module): 
    def __init__(self, in_channels, kernel_size=3):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(  
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels) 
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)    
        )
  
    def forward(self, x):    
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
 
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size() 
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)    
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x    

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    
     
    module = EUCB(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)