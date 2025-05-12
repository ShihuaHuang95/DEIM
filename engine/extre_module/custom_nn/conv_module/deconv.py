''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IEEETIP2024-DEConv.png    
论文链接：https://arxiv.org/pdf/2301.04805     
'''
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')     
from calflops import calculate_flops
   
import math
import torch 
from torch import nn
from einops.layers.torch import Rearrange
 
from engine.extre_module.ultralytics_nn.conv import Conv  
from engine.extre_module.torch_utils import model_fuse_test    

class Conv2d_cd(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):    
   
        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)    
        self.theta = theta
   
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        if conv_weight.is_cuda:
            conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)  
        else:
            conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)     
        conv_weight_cd = conv_weight_cd.to(conv_weight.dtype)    
        conv_weight_cd[:, :, :] = conv_weight[:, :, :] 
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias

  
class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):    
   
        super(Conv2d_ad, self).__init__()     
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta   
  
    def get_weight(self):  
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape    
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)   
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]     
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad) 
        return conv_weight_ad, self.conv.bias     


class Conv2d_rd(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,    
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):     
  
        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal 
        else:
            conv_weight = self.conv.weight 
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)  
            conv_weight_rd = conv_weight_rd.to(conv_weight.dtype)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight) 
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]     
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta  
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta) 
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)   
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups) 
    
            return out_diff  

 
class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):     

        super(Conv2d_hd, self).__init__()   
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)    

    def get_weight(self):    
        conv_weight = self.conv.weight     
        conv_shape = conv_weight.shape 
        if conv_weight.is_cuda:
            conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd = conv_weight_hd.to(conv_weight.dtype)     
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias
   
  
class Conv2d_vd(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,    
                 padding=1, dilation=1, groups=1, bias=False): 

        super(Conv2d_vd, self).__init__()   
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)   
     
    def get_weight(self):   
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        if conv_weight.is_cuda: 
            conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd = conv_weight_vd.to(conv_weight.dtype)    
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]   
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias  

     
class DEConv(nn.Module):  
    def __init__(self, inc, ouc):
        super(DEConv, self).__init__()   
        self.conv1_1 = Conv2d_cd(inc, inc, 3, bias=True)    
        self.conv1_2 = Conv2d_hd(inc, inc, 3, bias=True)
        self.conv1_3 = Conv2d_vd(inc, inc, 3, bias=True) 
        self.conv1_4 = Conv2d_ad(inc, inc, 3, bias=True)    
        self.conv1_5 = nn.Conv2d(inc, inc, 3, padding=1, bias=True)
        
        self.bn = nn.BatchNorm2d(inc)
        self.act = nn.SiLU()    

        if inc != ouc:     
            self.conv1x1 = Conv(inc, ouc, 1)
        else:
            self.conv1x1 = nn.Identity()     
    
    def forward(self, x):    
        if hasattr(self, 'conv1_1'):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight() 
            w3, b3 = self.conv1_3.get_weight()
            w4, b4 = self.conv1_4.get_weight()     
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias
    
            w = w1 + w2 + w3 + w4 + w5     
            b = b1 + b2 + b3 + b4 + b5
            res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        else:
            res = self.conv1_5(x)    
            
        if hasattr(self, 'bn'):
            res = self.bn(res)
  
        return self.conv1x1(self.act(res))
    
    def convert_to_deploy(self):  
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight() 
        w3, b3 = self.conv1_3.get_weight()     
        w4, b4 = self.conv1_4.get_weight()     
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias  

        self.conv1_5.weight = torch.nn.Parameter(w1 + w2 + w3 + w4 + w5)     
        self.conv1_5.bias = torch.nn.Parameter(b1 + b2 + b3 + b4 + b5)
  
        del self.conv1_1     
        del self.conv1_2     
        del self.conv1_3 
        del self.conv1_4
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     
    
    module = DEConv(in_channel, out_channel).to(device)   
  
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(GREEN + 'test reparameterization.' + RESET) 
    module = model_fuse_test(module)
    outputs = module(inputs)    
    print(GREEN + 'test reparameterization done.' + RESET)  
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
