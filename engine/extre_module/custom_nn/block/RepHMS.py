'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/RepHMS.png
论文链接：https://arxiv.org/abs/2502.04656  
'''   
    
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
 
import warnings 
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 
    
import torch
import torch.nn as nn     
import torch.nn.functional as F
import numpy as np
  
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.torch_utils import model_fuse_test  
     
class AVG(nn.Module):   
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])    

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)  
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)     
        return x   
     
class UniRepLKNetBlock(nn.Module):
    
    def __init__(self,     
                 dim,
                 kernel_size,     
                 deploy=False,
                 attempt_use_lk_impl=True): 
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')    
        if kernel_size == 0: 
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:    
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)   
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,     
                                     dilation=1, groups=dim, bias=deploy,  
                                     attempt_use_lk_impl=attempt_use_lk_impl)     
   
        if deploy or kernel_size == 0:     
            self.norm = nn.Identity()   
        else:
            self.norm = get_bn(dim)     
    
    
    def forward(self, inputs):
   
        out = self.norm(self.dwconv(inputs))     
        return out

    def convert_to_deploy(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()     
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (     
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std   
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)     
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1) 
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std    
                self.dwconv = conv
            self.norm = nn.Identity()
  
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)    
    We assume the inputs to this block are (N, C, H, W)    
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True): 
        super().__init__()  
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,     
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy, 
                                    )
        self.attempt_use_lk_impl = attempt_use_lk_impl  

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]    
            self.dilates = [1, 2, 4, 5, 7]    
        elif kernel_size == 15:     
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7] 
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]     
            self.dilates = [1, 1, 1] 
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1] 
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]  


        else:   
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r), 
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels, 
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))  

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))   
        return out
 
    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)  
            for k, r in zip(self.kernel_sizes, self.dilates):    
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,     
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)  
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):    
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

from itertools import repeat
import collections.abc
def _ntuple(n):     
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):     
            return tuple(x)
        return tuple(repeat(x, n))  
    return parse     
  
    
to_1tuple = _ntuple(1) 
to_2tuple = _ntuple(2) 
to_3tuple = _ntuple(3)  
to_4tuple = _ntuple(4)     
to_ntuple = _ntuple
def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()   
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,     
               attempt_use_lk_impl=True):   
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)   
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)    

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)    
def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device =kernel.device )
    if kernel.size(1) == 1:
        #   This is a DW kernel   
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate) 
        return dilated     
    else:
        #   This is a dense or group-wise (but not DW) kernel   
        slices = []
        for i in range(kernel.size(1)): 
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)     
            slices.append(dilated)  
        return torch.cat(slices, dim=1)
    
def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2) 
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2  
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel   
def get_bn(channels): 
    return nn.BatchNorm2d(channels) 
 
class DepthBottleneckUniv2(nn.Module):
    def __init__(self,
                 in_channels,     
                 out_channels,     
                 shortcut=True,
                 kersize=5,     
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):    
        super(DepthBottleneckUniv2, self).__init__()   

        mid_channel = int(in_channels * expansion_depth)    
        mid_channel2 = mid_channel  
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:     
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, 1) 

            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, 1) 
        else:
            self.conv2 = Conv(out_channels, out_channels, 3)

    def forward(self, x):    
        y = self.conv1(x)  
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))     
        y = self.one_conv2(y)     
        return y

class RepHMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True, 
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):    
        super(RepHMS, self).__init__() 
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width  
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1)  
        self.RepElanMSBlock = nn.ModuleList()   
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([  
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)     
                for _ in range(depth)     
            ])   
            self.RepElanMSBlock.append(DepthBlock)  
   
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1)     
     
    def forward(self, x):
        x = self.conv1(x)    
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]     
        x_out[1] = x_out[1] + x_out[0]     
        cascade = []   
        elan = [x_out[0]]  
        for i in range(self.width - 1):
            for j in range(self.depth):   
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:    
                        #cascade = [cascade[-1]]  
                        if self.depth > 1:
                            cascade =[cascade[-1]]
                        else:
                            cascade = []     
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:    
                    cascade.append(x_out[i + 1])
  
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)     
        return y_out   
   
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
   
    module = RepHMS(in_channel, out_channel, width=3, depth=1).to(device)

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