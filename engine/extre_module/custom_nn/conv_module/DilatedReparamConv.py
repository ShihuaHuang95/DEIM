'''  
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/CVPR2024-DilatedReparamConv.png     
论文链接：https://arxiv.org/abs/2311.15599 
'''    
   
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch
import torch.nn as nn  
import torch.nn.functional as F   
from timm.layers import to_2tuple
   
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.torch_utils import model_fuse_test

#================== This function decides which conv implementation (the native or iGEMM) to use
#   Note that iGEMM large-kernel conv impl will be used if 
#       -   you attempt to do so (attempt_to_use_large_impl=True), and 
#       -   it has been installed (follow https://github.com/AILab-CVC/UniRepLKNet), and
#       -   the conv layer is depth-wise, stride = 1, non-dilated, kernel_size > 5, and padding == kernel_size // 2
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)    
    else:
        padding = to_2tuple(padding)  
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)     

    if attempt_use_lk_impl and need_large_impl:     
        # print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM     
            # print('---------------- found iGEMM implementation ')
        except: 
            DepthWiseConv2dImplicitGEMM = None
            # print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:     
            # print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')  
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)    
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias) 
   
def get_bn(dim, use_sync_bn=False): 
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:     
        return nn.BatchNorm2d(dim)  

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias 
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):   
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device) 
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

class DilatedReparamConv(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)   
    We assume the inputs to this block are (N, C, H, W)
    """     
    def __init__(self, in_channels, out_channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(out_channels, out_channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=out_channels, bias=deploy,     
                                    attempt_use_lk_impl=attempt_use_lk_impl)     
        self.attempt_use_lk_impl = attempt_use_lk_impl
 
        if in_channels != out_channels:
            self.conv1x1 = Conv(in_channels, out_channels, k=1) # 用作调整通道数
        else:
            self.conv1x1 = nn.Identity()    

        #   Default settings. We did not tune them carefully. Different settings may work better.   
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
            self.kernel_sizes = [5, 5, 3, 3]   
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:     
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')
  
        if not deploy:
            self.origin_bn = get_bn(out_channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):   
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=out_channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(out_channels, use_sync_bn=use_sync_bn))

    def forward(self, x):     
        x = self.conv1x1(x)     
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))     
            out = out + bn(conv(x))    
        return out  

    def convert_to_deploy(self):
        if hasattr(self, 'origin_bn'): 
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))    
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))     
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r) 
                origin_b += branch_b    
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv    
            self.__delattr__('origin_bn')  
            for k, r in zip(self.kernel_sizes, self.dilates):     
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))    
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))  

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = DilatedReparamConv(in_channel, out_channel, kernel_size=11).to(device)    
  
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