'''    
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2025-EfficientVIM.png
论文链接：https://arxiv.org/abs/2411.15241 
论文链接：https://arxiv.org/abs/2311.17132     
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops 

import torch
import torch.nn as nn
  
from engine.extre_module.ultralytics_nn.conv import Conv   

class LayerNorm2D(nn.Module):    
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):    
        super(LayerNorm2D, self).__init__()    
        self.num_channels = num_channels    
        self.eps = eps
        self.affine = affine
   
        if self.affine: 
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))     
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)    
 
    def forward(self, x):     
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W) 
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)
   
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)  
  
        if self.affine:  
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized
     

class LayerNorm1D(nn.Module):     
    """LayerNorm for channels of 1D tensor(B C L)"""   
    def __init__(self, num_channels, eps=1e-5, affine=True):  
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels   
        self.eps = eps     
        self.affine = affine

        if self.affine:   
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1)) 
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))   
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None) 
 
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)
    
        if self.affine:     
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized
    
   
class ConvLayer2D(nn.Module):   
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):    
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim, 
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,   
            bias=False    
        ) 
        self.norm = norm(num_features=out_dim) if norm else None 
        self.act = act_layer() if act_layer else None
  
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)   
            torch.nn.init.constant_(self.norm.bias, 0)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        x = self.conv(x)
        if self.norm: 
            x = self.norm(x)
        if self.act:     
            x = self.act(x) 
        return x
    
    
class ConvLayer1D(nn.Module):     
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__() 
        self.conv = nn.Conv1d(  
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  
            dilation=dilation,  
            groups=groups,  
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None    
        self.act = act_layer() if act_layer else None  
     
        if self.norm:   
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        x = self.conv(x)  
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)   
        return x
    

class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()     
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)   
        
    def forward(self, x):  
        x = self.fc2(self.fc1(x))
        return x  
   
class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):   
        super().__init__() 
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None) 
        conv_dim = self.state_dim*3   
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)    
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)     
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()     
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True
     
    def forward(self, x, size):     
        batch, _, L= x.shape 
        
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, size[0], size[1])).flatten(2)
        B,C,dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1)   
        A = (dt.contiguous() + self.A.view(1,-1,1)).softmax(-1)   
 
        AB = (A * B.contiguous())   
        h = x @ AB.transpose(-2,-1)     
        
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h.contiguous() * self.act(z.contiguous())+ h.contiguous() * self.D)
        y = h @ C.contiguous() # B C N, B C L -> B C L    
        
        y = y.view(batch,-1, size[0], size[1]).contiguous()# + x * self.D  # B C H W 
        return y, h  

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
     
class EfficientViMBlock(nn.Module):  
    def __init__(self, inc, ouc, mlp_ratio=4., ssd_expand=1, state_dim=64):    
        super().__init__()
        self.dim = inc  
        self.mlp_ratio = mlp_ratio
        
        self.mixer = HSMSSD(d_model=inc, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(inc)
        
        self.dwconv1 = ConvLayer2D(inc, inc, 3, padding=1, groups=inc, bn_weight_init=0, act_layer = None)  
        self.dwconv2 = ConvLayer2D(inc, inc, 3, padding=1, groups=inc, bn_weight_init=0, act_layer = None)
  
        self.ffn = FFN(in_dim=inc, dim=int(inc * mlp_ratio))
        
        #LayerScale   
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, inc), requires_grad=True)     

        if inc != ouc:    
            self.conv1x1 = Conv(inc, ouc)
        else:
            self.conv1x1 = nn.Identity()   
        
    def forward(self, x):  
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)     
   
        # DWconv1
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)  
        
        # HSM-SSD   
        x_prev = x
        _, _, H, W = x.size()  
        x, h = self.mixer(self.norm(x.flatten(2)), (H, W))    
        x = (1-alpha[1]) * x_prev + alpha[1] * x  
        
        # DWConv2   
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
   
        # FFN
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)

        return self.conv1x1(x) 

class EfficientViMBlock_CGLU(EfficientViMBlock):
    def __init__(self, inc, ouc, mlp_ratio=4, ssd_expand=1, state_dim=64):     
        super().__init__(inc, ouc, mlp_ratio, ssd_expand, state_dim)    
  
        self.ffn = ConvolutionalGLU(inc, hidden_features=int(inc * mlp_ratio))

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
     
    print(RED + '-'*20 + " EfficientViMBlock " + '-'*20 + RESET)

    module = EfficientViMBlock(in_channel, out_channel).to(device)   
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True) 
    print(RESET)  

    print(RED + '-'*20 + " EfficientViMBlock_CGLU " + '-'*20 + RESET)    

    module = EfficientViMBlock_CGLU(in_channel, out_channel).to(device) 
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,     
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)   
