'''  
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/IEEETIP2020-ContextGuidedBlock_Down.png   
论文链接：https://arxiv.org/pdf/1811.08201
'''
     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
     
import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops     
   
import torch 
import torch.nn as nn    
from engine.extre_module.ultralytics_nn.conv import Conv, autopad 
    
class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.    
    """     
    def __init__(self, channel, reduction=16):   
        super(FGlo, self).__init__()     
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid() 
        )
  
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock(nn.Module): 
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels  
           nOut: number of output channels, 
           add: if true, residual learning 
        """    
        super().__init__() 
        n= int(nOut/2) 
        self.conv1x1 = Conv(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)   
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=n) # surrounding context 
        self.bn_act = nn.Sequential(    
            nn.BatchNorm2d(nOut),    
            Conv.default_act
        )
        self.add = add  
        self.F_glo= FGlo(nOut, reduction)  
    
    def forward(self, input):
        output = self.conv1x1(input) 
        loc = self.F_loc(output)   
        sur = self.F_sur(output) 
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_act(joi_feat)   

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature   
        # if residual version  
        if self.add:
            output  = input + output 
        return output  

class ContextGuidedBlock_Down(nn.Module):     
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map     
           nOut: the channel of output feature map  
        """   
        super().__init__()  
        self.conv1x1 = Conv(nIn, nOut, 3, s=2)  #  size/2, channel: nIn--->nOut  
        
        self.F_loc = nn.Conv2d(nOut, nOut, 3, padding=1, groups=nOut)    
        self.F_sur = nn.Conv2d(nOut, nOut, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=nOut) 
        
        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = Conv.default_act 
        self.reduce = Conv(2 * nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
     
        self.F_glo = FGlo(nOut, reduction)  
  
    def forward(self, input):
        print(111)
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)     

        joi_feat = torch.cat([loc, sur],1)  #  the joint feature
        joi_feat = self.bn(joi_feat)    
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut 
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature   

        return output
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 

    module = ContextGuidedBlock_Down(in_channel, out_channel).to(device)
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)   
    print(RESET)