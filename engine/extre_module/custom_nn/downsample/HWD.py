'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/HWD.png
论文链接：https://www.sciencedirect.com/science/article/pii/S0031320323005174
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings    
warnings.filterwarnings('ignore')  
from calflops import calculate_flops    
     
import torch
import torch.nn as nn
from engine.extre_module.ultralytics_nn.conv import Conv
    
class HWD(nn.Module):
    def __init__(self, in_ch, out_ch): 
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')   
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)
 
    def forward(self, x):
        yL, yH = self.wt(x)    
        y_HL = yH[0][:,:,0,::]   
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.conv(x)
   
        return x
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = HWD(in_channel, out_channel).to(device)
   
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)