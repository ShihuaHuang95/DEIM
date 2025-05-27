'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2023-FasterBlock.png
论文链接：https://arxiv.org/pdf/2303.03667
论文链接：https://arxiv.org/abs/2311.17132
'''    
 
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    

import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops 

import torch
import torch.nn as nn
from timm.layers import DropPath

from engine.extre_module.ultralytics_nn.conv import Conv    
     
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):    
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3 
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  

        if forward == 'slicing':    
            self.forward = self.forward_slicing
        elif forward == 'split_cat': 
            self.forward = self.forward_split_cat   
        else:
            raise NotImplementedError
 
    def forward_slicing(self, x):     
        # only for inference   
        x = x.clone()   # !!! Keep the original input intact for the residual connection later   
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x 
  
    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)    
        x = torch.cat((x1, x2), 1)     
        return x   

class Faster_Block(nn.Module):     
    def __init__(self,     
                 inc,
                 ouc, 
                 n_div=4,   
                 mlp_ratio=2,
                 drop_path=0.1,    
                 layer_scale_init_value=0.0,     
                 pconv_fw_type='split_cat',
                 act = None
                 ):     
        super().__init__()
        self.ouc = ouc
        self.mlp_ratio = mlp_ratio  
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()     
        self.n_div = n_div

        mlp_hidden_dim = int(ouc * mlp_ratio)   
   
        mlp_layer = [ 
            Conv(ouc, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, ouc, 1, bias=False)
        ] 

        self.mlp = nn.Sequential(*mlp_layer)
 
        self.spatial_mixing = Partial_conv3(
            ouc,
            n_div,    
            pconv_fw_type   
        )     
        
        self.adjust_channel = None
        if inc != ouc:    
            self.adjust_channel = Conv(inc, ouc, 1)     

        if layer_scale_init_value > 0:  
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((ouc)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:     
            self.forward = self.forward
     
    def forward(self, x):  
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x    
        x = self.spatial_mixing(x) 
        x = shortcut + self.drop_path(self.mlp(x)) 
        return x
     
    def forward_layer_scale(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)   
        x = shortcut + self.drop_path( 
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))   
        return x 

class ConvolutionalGLU(nn.Module):     
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()  
        out_features = out_features or in_features  
        hidden_features = hidden_features or in_features 
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(   
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features), 
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  
        self.drop = nn.Dropout(drop)

    def forward(self, x):   
        x_shortcut = x     
        x, v = self.fc1(x).chunk(2, dim=1)    
        x = self.dwconv(x) * v
        x = self.drop(x)     
        x = self.fc2(x)  
        x = self.drop(x)
        return x_shortcut + x

class Faster_Block_CGLU(Faster_Block): 
    def __init__(self, inc, ouc, n_div=4, mlp_ratio=2, drop_path=0.1, layer_scale_init_value=0, pconv_fw_type='split_cat'):
        super().__init__(inc, ouc, n_div, mlp_ratio, drop_path, layer_scale_init_value, pconv_fw_type)   
        self.mlp = ConvolutionalGLU(ouc)  
  
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    print(RED + '-'*20 + " Faster_Block " + '-'*20 + RESET)

    module = Faster_Block(in_channel, out_channel, n_div=4).to(device)

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)  
  
    print(RED + '-'*20 + " Faster_Block_CGLU " + '-'*20 + RESET)  

    module = Faster_Block_CGLU(in_channel, out_channel, n_div=4).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
