''' 
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/SEAM.png  
论文链接：https://arxiv.org/pdf/2208.02019v2
'''
    
import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import torch, math
import torch.nn as nn
import torch.nn.functional as F     
   
class Residual(nn.Module):    
    def __init__(self, fn): 
        super(Residual, self).__init__()    
        self.fn = fn
 
    def forward(self, x):
        return self.fn(x) + x     
     
class SEAM(nn.Module):
    def __init__(self, c1, n=1, reduction=16):  
        super(SEAM, self).__init__()
        self.DCovN = nn.Sequential(  
            *[nn.Sequential(
                Residual(nn.Sequential(  
                    nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1, groups=c1),    
                    nn.GELU(),
                    nn.BatchNorm2d(c1)
                )),
                nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=1, stride=1, padding=0, groups=1), 
                nn.GELU(),  
                nn.BatchNorm2d(c1)
            ) for i in range(n)]   
        )     
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)    
        self.fc = nn.Sequential( 
            nn.Linear(c1, c1 // reduction, bias=False),    
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid()   
        )

        self._initialize_weights()   
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)    
 
  
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)  
        y = self.avg_pool(y).view(b, c)    
        y = self.fc(y).view(b, c, 1, 1)     
        y = torch.exp(y)
        return x * y.expand_as(x)
 
    def _initialize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    

    def initialize_layer(self, layer):  
        if isinstance(layer, (nn.Conv2d, nn.Linear)):     
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:  
                torch.nn.init.constant_(layer.bias, 0)    
  
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel, height, width = 1, 16, 32, 32  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     
  
    module = SEAM(channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)