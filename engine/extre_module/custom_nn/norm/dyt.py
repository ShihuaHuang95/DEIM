'''     
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2025-Dynamic Tanh.png    
论文链接：https://arxiv.org/abs/2503.10622
'''
  
import warnings    
warnings.filterwarnings('ignore')   
from calflops import calculate_flops

import torch     
import torch.nn as nn

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):    
        super().__init__()
        self.normalized_shape = normalized_shape    
        self.alpha_init_value = alpha_init_value    
        self.channels_last = channels_last
     
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)  
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:    
            x = x * self.weight + self.bias
        else:  
            x = x * self.weight[:, None, None] + self.bias[:, None, None]    
        return x   

    def extra_repr(self):     
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"     
     
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     
     
    module = DynamicTanh(channel).to(device) 

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)    
