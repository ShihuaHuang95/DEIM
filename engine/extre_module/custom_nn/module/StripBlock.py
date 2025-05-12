'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/StripBlock.png
论文链接：https://arxiv.org/pdf/2501.03775
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
     
import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops   
  
import torch
import torch.nn as nn
from timm.layers import DropPath  

from engine.extre_module.ultralytics_nn.conv import Conv, DWConv   

class StripMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features   
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)    
        self.dwconv = DWConv(hidden_features, hidden_features) 
        self.act = act_layer()     
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Strip_Block(nn.Module):   
    def __init__(self, dim, k1, k2):     
        super().__init__()     
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
  
        self.conv1 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x): 
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)  
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return x * attn
   
class Strip_Attention(nn.Module):    
    def __init__(self, d_model,k1,k2):   
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)    
        self.activation = nn.GELU()
        self.spatial_gating_unit = Strip_Block(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):    
        shorcut = x.clone() 
        x = self.proj_1(x)
        x = self.activation(x)
        # x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut    
        return x

class StripBlock(nn.Module):  
    def __init__(self, inc, dim, mlp_ratio=4., k1=1, k2=19, drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Strip_Attention(dim, k1, k2)     
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = StripMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)    
        layer_scale_init_value = 1e-2    
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter( 
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)   

        self.conv1x1 = Conv(inc, dim, k=1) if inc != dim else nn.Identity()    
  
    def forward(self, x):  
        x = self.conv1x1(x) 
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))   
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    module = StripBlock(in_channel, out_channel).to(device) 

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
