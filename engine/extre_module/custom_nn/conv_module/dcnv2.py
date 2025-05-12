'''
本文件由BiliBili：魔傀面具整理 
论文链接：https://arxiv.org/pdf/1811.11168   
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch, math
import torch.nn as nn 

from engine.extre_module.ultralytics_nn.conv import Conv, autopad   

# Deformable Conv v2
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,    
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):  
        super(DCNv2, self).__init__()   

        self.in_channels = in_channels   
        self.out_channels = out_channels    
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)  
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding) 
        self.dilation = (dilation, dilation)  
        self.groups = groups
        self.deformable_groups = deformable_groups    
    
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,    
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,    
            bias=True,
        )     
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()     

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)  
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)  
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x, 
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups, 
            True
        )
        x = self.bn(x)  
        x = self.act(x)
        return x     

    def reset_parameters(self):
        n = self.in_channels   
        for k in self.kernel_size:
            n *= k    
        std = 1. / math.sqrt(n)  
        self.weight.data.uniform_(-std, std)   
        self.bias.data.zero_()  
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
   
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     
   
    module = DCNv2(in_channel, out_channel, kernel_size=3, stride=1).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)