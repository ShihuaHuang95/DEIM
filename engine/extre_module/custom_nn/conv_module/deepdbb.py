'''
本文件由BiliBili：魔傀面具整理
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S1474034624003574
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
from engine.extre_module.ultralytics_nn.conv import Conv, autopad

from engine.extre_module.torch_utils import model_fuse_test
    
def transI_fusebn(kernel, bn):
    gamma = bn.weight 
    std = (bn.running_var + bn.eps).sqrt()  
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std  
 
def transII_addbranch(kernels, biases): 
    return sum(kernels), sum(biases)
  
def transIII_1x1_kxk(k1, b1, k2, b2, groups):     
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      # 
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))    
    else:   
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :] 
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]     
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))    
        k, b_hat = transIV_depthconcat(k_slices, b_slices)     
    return k, b_hat + b2
  
def transIV_depthconcat(kernels, biases):     
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups   
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))     
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2    
    return k   

#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels  
def transVI_multiscale(kernel, target_kernel_size):  
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2   
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2   
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):  
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups, 
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)  
    se.add_module('bn', bn_layer)    
    return se


class IdentityBasedConv1x1(nn.Module):  
    def __init__(self, channels, groups=1):
        super().__init__()
        assert channels % groups == 0
        input_dim = channels // groups
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups, bias=False) 
        
        id_value = np.zeros((channels, input_dim, 1, 1))  
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value)    
        nn.init.zeros_(self.conv.weight)
        self.groups = groups  
   
    def forward(self, input):
        kernel = self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)
        result = F.conv2d(input, kernel, None, stride=1, groups=self.groups)
        return result  

    def get_actual_kernel(self):  
        return self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)  
    
class BNAndPadLayer(nn.Module):  
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,   
                 momentum=0.1,
                 affine=True,    
                 track_running_stats=True):  
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) 
        self.pad_pixels = pad_pixels
    
    def forward(self, input):
        output = self.bn(input)  
        if self.pad_pixels > 0:
            if self.bn.affine:     
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)     
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)   
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values    
            output[:, :, :, 0:self.pad_pixels] = pad_values   
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output
     
    @property    
    def weight(self):
        return self.bn.weight  

    @property
    def bias(self):
        return self.bn.bias
     
    @property  
    def running_mean(self):
        return self.bn.running_mean     

    @property
    def running_var(self):
        return self.bn.running_var  
  
    @property     
    def eps(self):
        return self.bn.eps 
     
class DiverseBranchBlockNOAct(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1, 
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlockNOAct, self).__init__()    
        self.deploy = deploy

        # self.nonlinear = Conv.default_act 

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups     
    
        if padding is None:
            # padding=None
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,     
                                         padding=padding, dilation=dilation, groups=groups, bias=True) 

        else: 

            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)    

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:   
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,  
                                                  stride=1, padding=0, groups=groups, bias=False)) 
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))  
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,    
                                       padding=0, groups=groups)   
            else: 
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))
 
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels   

            self.dbb_1x1_kxk = nn.Sequential()  
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))   
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))
            self.dbb_1x1_kxk.add_module('conv2',    
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels, 
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,  
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))    
 
        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.   
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):  
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:    
            k_1x1, b_1x1 = 0, 0    

        if hasattr(self.dbb_1x1_kxk, 'idconv1'): 
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()   
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1) 
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)   
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,  
                                                              b_1x1_kxk_second, groups=self.groups)
 
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)  
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):   
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn) 
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,   
                                                                  b_1x1_avg_second, groups=self.groups)
        else:  
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second  

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):   
            return   
        kernel, bias = self.get_equivalent_kernel_bias()  
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, 
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,    
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel     
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():    
            para.detach_()   
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')  
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')   
        self.__delattr__('dbb_1x1_kxk')     

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam(inputs)  
  
        out = self.dbb_origin(inputs)
   
        # print(inputs.shape)
        # print(self.dbb_1x1(inputs).shape)
        if hasattr(self, 'dbb_1x1'): 
            out += self.dbb_1x1(inputs)  
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)     
        # return self.nonlinear(out)  
    
        return out
     
    def init_gamma(self, gamma_value):     
        if hasattr(self, "dbb_origin"):     
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):  
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):    
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)   
   
    def single_init(self):    
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)     
 
    @property 
    def weight(self):  ##含有@property
        if hasattr(self, 'dbb_reparam'):
            # return self.nonlinear(self.dbb_reparam(inputs))
            return self.dbb_reparam.weight    

class DeepDiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False,conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlock, self).__init__()     
        self.deploy = deploy    
   
        self.nonlinear = Conv.default_act     

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups  
        # padding=0
        if padding is None:     
            padding = autopad(kernel_size, padding, dilation)     
        assert padding == kernel_size // 2 
     
        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True)   
    
        else:

            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()     
            if groups < out_channels:   
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))    
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))     
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)   
            else:     
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))   
  
            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels)) 

            if internal_channels_1x1_3x3 is None:    
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels
     
            self.dbb_1x1_kxk = nn.Sequential()   
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))     
            else:
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3, 
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True))    
            self.dbb_1x1_kxk.add_module('conv2',     
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,   
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels)) 

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.    
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.     
            self.single_init()    
     
    def get_equivalent_kernel_bias(self): 
        self.dbb_origin.switch_to_deploy()  
        # k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.dbb_reparam.weight, self.dbb_origin.bn)

        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias    
  
        if hasattr(self, 'dbb_1x1'):  
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)  
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:     
            k_1x1, b_1x1 = 0, 0
  
        if hasattr(self.dbb_1x1_kxk, 'idconv1'):    
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)  
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)
   
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)     
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second    

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),     
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))  

    def convert_to_deploy(self):    
        if hasattr(self, 'dbb_reparam'):  
            return     
        kernel, bias = self.get_equivalent_kernel_bias()     
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size, stride=self.dbb_origin.dbb_reparam.stride,  
                                     padding=self.dbb_origin.dbb_reparam.padding, dilation=self.dbb_origin.dbb_reparam.dilation, 
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')  
        self.__delattr__('dbb_avg')    
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')   
     
    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs)) 
            # return self.dbb_reparam(inputs)  

        out = self.dbb_origin(inputs) 
        if hasattr(self, 'dbb_1x1'):   
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs) 
        return self.nonlinear(out)     
    
        # return out    
   
    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"): 
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)     
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):   
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)  
     
    def single_init(self):
        self.init_gamma(0.0) 
        if hasattr(self, "dbb_origin"): 
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)
     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
   
    module = DeepDiverseBranchBlock(in_channel, out_channel, kernel_size=3, stride=1).to(device)
 
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
