'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2023-LSKBlock.png     
论文链接：https://arxiv.org/pdf/2403.11735
'''    

import warnings   
warnings.filterwarnings('ignore')    
from calflops import calculate_flops    

import torch
import torch.nn as nn  
 
class LSKBlock_SA(nn.Module):
    def __init__(self, dim): 
        super().__init__()  
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)   
        self.conv1 = nn.Conv2d(dim, dim//2, 1)    
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
     
    def forward(self, x):   
        attn1 = self.conv0(x)   
        attn2 = self.conv_spatial(attn1)   
  
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True) 
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)     
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)     
        attn = self.conv(attn) 
        return x * attn

class LSKBlock(nn.Module): 
    def __init__(self, d_model):  
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1) 
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKBlock_SA(d_model)  
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()     
        x = self.proj_1(x)    
        x = self.activation(x)     
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut   
        return x

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 16, 32, 32  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
 
    module = LSKBlock(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)