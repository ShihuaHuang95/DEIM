'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-ShiftwiseConv.png
论文链接：https://arxiv.org/abs/2401.12736
''' 

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
 
import warnings  
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
 
import math
import torch     
import torch.nn as nn
import torch.nn.functional as F   

from engine.extre_module.torch_utils import model_fuse_test    

def get_conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias  
):    
    # return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    try:
        paddings = (kernel_size[0] // 2, kernel_size[1] // 2)   
    except Exception as e:   
        paddings = padding
    return nn.Conv2d(   
        in_channels, out_channels, kernel_size, stride, paddings, dilation, groups, bias
    ) 

def get_bn(channels):  
    return nn.BatchNorm2d(channels)   
     
class Mask(nn.Module):
    def __init__(self, size):
        super().__init__()  
        self.weight = torch.nn.Parameter(data=torch.Tensor(*size), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
  
    def forward(self, x):
        w = torch.sigmoid(self.weight) 
        masked_wt = w.mul(x)    
        return masked_wt
   
def conv_bn_ori(  
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True    
):
    if padding is None:   
        padding = kernel_size // 2  
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,    
            kernel_size=kernel_size,
            stride=stride,    
            padding=padding,
            dilation=dilation,
            groups=groups,    
            bias=False,   
        ),
    )

    if bn:
        result.add_module("bn", get_bn(out_channels))     
    return result     

class LoRAConvsByWeight(nn.Module):   
    '''     
    merge LoRA1 LoRA2  
    shuffle channel by weights rather index
    ''' 
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int,
                 big_kernel, small_kernel,    
                 stride=1, group=1,   
                 bn=True, use_small_conv=True):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)   
        self.stride = stride
        self.small_conv = use_small_conv  
        # add same padding for vertical and horizon axis. should delete it accordingly    
        padding, after_padding_index, index = self.shift(self.kernels)
        self.pad = padding, after_padding_index, index
        self.nk = math.ceil(big_kernel / small_kernel)    
        out_n = out_channels * self.nk
        self.split_convs = nn.Conv2d(in_channels, out_n,
                                     kernel_size=small_kernel, stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)     

        self.lora1 = Mask((1, out_n, 1, 1))   
        self.lora2 = Mask((1, out_n, 1, 1))
        self.use_bn = bn 
 
        if bn:   
            self.bn_lora1 = get_bn(out_channels)     
            self.bn_lora2 = get_bn(out_channels)   
        else:
            self.bn_lora1 = None    
            self.bn_lora2 = None 

    def forward(self, inputs):
        out = self.split_convs(inputs)
        # split output
        *_, ori_h, ori_w = inputs.shape    
        lora1_x = self.forward_lora(self.lora1(out), ori_h, ori_w, VH='H', bn=self.bn_lora1)    
        lora2_x = self.forward_lora(self.lora2(out), ori_h, ori_w, VH='W', bn=self.bn_lora2)
        x = lora1_x + lora2_x 
        return x  
  
    def forward_lora(self, out, ori_h, ori_w, VH='H', bn=None):     
        # shift along the index of every group
        b, c, h, w = out.shape
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※   
        x = 0  
        for i in range(self.nk): 
            outi = self.rearrange_data(out[i], i, ori_h, ori_w, VH)  
            x = x + outi
        if self.use_bn:    
            x = bn(x)   
        return x    
     
    def rearrange_data(self, x, idx, ori_h, ori_w, VH):
        padding, _, index = self.pad  
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride    
        # need to calculate start point after conv
        # how many windows shift from real start window index 
        if (idx + 1) >= index:    
            pad_l = 0
            s = (idx + 1 - index) * (k // stride)    
        else: 
            pad_l = (index - 1 - idx) * (k // stride)
            s = 0  
        if VH == 'H':   
            # assume add sufficient padding for origin conv    
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l     
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            # e = w + pad_l + pad_r - s - suppose_len 
        else:    
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            # e = h + pad_l + pad_r - s - suppose_len 
        # print('new_pad', new_pad)  
        if len(set(new_pad)) > 1:     
            x = F.pad(x, new_pad)
        # split_list = [s, suppose_len, e]
        # padding on v direction 
        if padding * 2 + 1 != k:
            pad = padding - k // 2
            if VH == 'H':  # horizonal     
                x = torch.narrow(x, 2, pad, h - 2 * pad)
            else:  # vertical 
                x = torch.narrow(x, 3, pad, w - 2 * pad)

        xs = torch.narrow(x, dim, s, suppose_len)
        return xs    
     
    def shift(self, kernels):
        '''   
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.   
        '''
        mink, maxk = min(kernels), max(kernels)  
        mid_p = maxk // 2     
        # 1. new window size is mink. middle point index in the window
        offset_idx_left = mid_p % mink  
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink     
        # 2. padding
        padding = offset_idx_left % mink
        while padding < offset_idx_right:     
            padding += mink
        # 3. make sure last pixel can be scan by min window   
        while padding < (mink - 1):
            padding += mink
        # 4. index of windows start point of middle point  
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink  
        # 5. output:padding how to padding input in v&h direction;
        # after_padding_index: middle point of original kernel will located in which window  
        # real_start_idx: start window index after padding in original kernel along long side 
        return padding, after_padding_index, real_start_idx    


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True, use_small_conv=True): 
    if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:    
        return conv_bn_ori(  
            in_channels,     
            out_channels,
            kernel_size,
            stride,   
            padding,  
            groups,
            dilation,
            bn)
    else:     
        big_kernel, small_kernel = kernel_size    
        return LoRAConvsByWeight(in_channels, out_channels, bn=bn,
                                 big_kernel=big_kernel, small_kernel=small_kernel,  
                                 group=groups, stride=stride,    
                                 use_small_conv=use_small_conv)     

 
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

     
class ReparamLargeKernelConv(nn.Module):
    def __init__(
            self,  
            in_channels,    
            out_channels,    
            kernel_size,   
            small_kernel=5, 
            stride=1,  
            groups=1,
            small_kernel_merged=False,
            Decom=True,
            bn=True,
    ):
        super(ReparamLargeKernelConv, self).__init__()     
        self.kernel_size = kernel_size    
        self.small_kernel = small_kernel  
        self.Decom = Decom  
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.    
        padding = kernel_size // 2 
        if small_kernel_merged:  # cpp版本的conv，加快速度
            self.lkb_reparam = get_conv2d(  
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,    
                stride=stride, 
                padding=padding,  
                dilation=1,   
                groups=groups,     
                bias=True, 
            )
        else:  
            if self.Decom:
                self.LoRA = conv_bn(  
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, small_kernel),
                    stride=stride,
                    padding=padding,
                    dilation=1,     
                    groups=groups,  
                    bn=bn   
                )    
            else: 
                self.lkb_origin = conv_bn(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,  
                    stride=stride,    
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn, 
                )
    
            if (small_kernel is not None) and small_kernel < kernel_size:     
                self.small_conv = conv_bn(  
                    in_channels=in_channels,   
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,    
                    groups=groups,   
                    dilation=1,
                    bn=bn,
                )
  
        self.bn = get_bn(out_channels)
        self.act = nn.SiLU()
     
    def forward(self, inputs):  
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            # out = self.LoRA1(inputs) + self.LoRA2(inputs)     
            out = self.LoRA(inputs)
            if hasattr(self, "small_conv"):     
                out += self.small_conv(inputs)  
        else:    
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return self.act(self.bn(out)) 

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn) 
        if hasattr(self, "small_conv"):   
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part  
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def convert_to_deploy(self):
        if hasattr(self, 'lkb_origin'):    
            eq_k, eq_b = self.get_equivalent_kernel_bias()
            self.lkb_reparam = get_conv2d(   
                in_channels=self.lkb_origin.conv.in_channels, 
                out_channels=self.lkb_origin.conv.out_channels,
                kernel_size=self.lkb_origin.conv.kernel_size,    
                stride=self.lkb_origin.conv.stride,   
                padding=self.lkb_origin.conv.padding,    
                dilation=self.lkb_origin.conv.dilation,
                groups=self.lkb_origin.conv.groups,
                bias=True,   
            )
            self.lkb_reparam.weight.data = eq_k
            self.lkb_reparam.bias.data = eq_b 
            self.__delattr__("lkb_origin") 
            if hasattr(self, "small_conv"):   
                self.__delattr__("small_conv") 

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
    
    module = ReparamLargeKernelConv(in_channel, out_channel, kernel_size=13).to(device)

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
