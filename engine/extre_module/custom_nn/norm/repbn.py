''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICML2024-RepBN.png
论文链接：https://arxiv.org/pdf/2405.11582    
'''     
 
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
     
import torch
import torch.nn as nn   

class RepBN(nn.Module):   
    def __init__(self, channels):    
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)     

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)     
        return x

class LinearNorm(nn.Module):
    def __init__(self, dim, norm1=nn.LayerNorm, norm2=RepBN, warm=0, step=10000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('warm', torch.tensor(warm))    
        self.register_buffer('iter', torch.tensor(step))   
        self.register_buffer('total_step', torch.tensor(step))  
        self.r0 = r0
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)   
 
    def forward(self, x):
        if self.training:
            if self.warm > 0:     
                self.warm.copy_(self.warm - 1)   
                x = self.norm1(x)
            else: 
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)     
                x2 = self.norm2(x)     
                x = lamda * x1 + (1 - lamda) * x2 
        else:   
            x = self.norm2(x)
        return x

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, height * width, channel)).to(device)

    module = LinearNorm(channel).to(device)  
  
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, height * width, channel), 
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET) 
