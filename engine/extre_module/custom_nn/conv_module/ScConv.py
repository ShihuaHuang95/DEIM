'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2023-SCConv.png 
论文链接：https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf 
'''  

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch
import torch.nn as nn    
import torch.nn.functional as F  
   
from engine.extre_module.ultralytics_nn.conv import Conv 
 
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16,    
                 eps:float = 1e-10 
                 ):
        super(GroupBatchnorm2d,self).__init__()  
        assert c_num    >= group_num
        self.group_num  = group_num 
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1)) 
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps        = eps

    def forward(self, x):   
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )   
        mean        = x.mean(   dim = 2, keepdim = True ) 
        std         = x.std (   dim = 2, keepdim = True )   
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta   
  
class SRU(nn.Module):
    def __init__(self,     
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5    
                 ):
        super().__init__() 
        
        self.gn             = GroupBatchnorm2d( oup_channels, group_num = group_num )     
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()  
 
    def forward(self,x):
        gn_x        = self.gn(x)   
        w_gamma     = self.gn.gamma/sum(self.gn.gamma)  
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = reweigts>=self.gate_treshold     
        noninfo_mask= reweigts<self.gate_treshold  
        x_1         = info_mask * x
        x_2         = noninfo_mask * x
        x           = self.reconstruct(x_1,x_2)
        return x
    
    def reconstruct(self,x_1,x_2):     
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)     
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)

  
class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    ''' 
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,    
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,    
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel 
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)    
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)   
        #up 
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size) 
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low 
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)   
        self.advavg         = nn.AdaptiveAvgPool2d(1)
   
    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)    
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform 
        Y1      = self.GWC(up) + self.PWC1(up) 
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )  
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )  
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)   
        return out1+out2

    
class ScConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py     
    def __init__(self,     
                in_channel:int,     
                out_channel:int,   
                group_num:int = 16,   
                gate_treshold:float = 0.5,  
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,  
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):   
        super().__init__()
        self.SRU = SRU(in_channel, 
                       group_num            = group_num,     
                       gate_treshold        = gate_treshold)    
        self.CRU = CRU(in_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size)     

        if in_channel != out_channel:     
            self.conv1x1 = Conv(in_channel, out_channel, 1)  
        else:    
            self.conv1x1 = nn.Identity()
    
    def forward(self,x):  
        x = self.SRU(x)
        x = self.CRU(x)   
        return self.conv1x1(x)   

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = ScConv(in_channel, out_channel).to(device)
  
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)