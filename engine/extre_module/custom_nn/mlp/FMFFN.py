'''   
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/ICLR2024-FMFFN.png
论文链接：https://arxiv.org/pdf/2310.16387
'''   

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch     
import torch.nn as nn
from einops import rearrange 

class WindowFrequencyModulation(nn.Module): 
    def __init__(self, dim, window_size):
        super().__init__() 
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1    
        self.complex_weight= nn.Parameter(torch.cat((torch.ones(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32),\
        torch.zeros(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32)),dim=-1))     
 
    def forward(self, x):
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)    
  
        x = x.to(torch.float32)    
        
        x= torch.fft.rfft2(x,dim=(3, 4), norm='ortho')
     
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')  

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')
        return x    

class FMFFN(nn.Module):
    def __init__(self,in_features, hidden_features=None, out_features=None, window_size=4, act_layer=nn.GELU) -> None:   
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features

        self.ffn = nn.Sequential(   
            nn.Conv2d(in_features, hidden_features, 1),    
            act_layer(),   
            nn.Conv2d(hidden_features, out_features, 1)  
        )
  
        self.fm = WindowFrequencyModulation(out_features, window_size)     
    
    def forward(self, x):    
        return self.fm(self.ffn(x))

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = FMFFN(in_features=in_channel, out_features=out_channel).to(device)
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,     
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)