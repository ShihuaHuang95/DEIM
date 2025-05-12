'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/AAAI2024-DynamicFilter.png
论文链接：https://arxiv.org/pdf/2303.03932
''' 
     
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
from timm.layers import to_2tuple

from engine.extre_module.ultralytics_nn.conv import Conv 

def resize_complex_weight(origin_weight, new_h, new_w): 
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2   
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)  
    new_weight = torch.nn.functional.interpolate( 
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2) 
    return new_weight    
     
class StarReLU(nn.Module):  
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,   
                 mode=None, inplace=False):    
        super().__init__() 
        self.inplace = inplace    
        self.relu = nn.ReLU(inplace=inplace)  
        self.scale = nn.Parameter(scale_value * torch.ones(1),  
                                  requires_grad=scale_learnable) 
        self.bias = nn.Parameter(bias_value * torch.ones(1),  
                                 requires_grad=bias_learnable)  
  
    def forward(self, x): 
        return self.scale * self.relu(x) ** 2 + self.bias    
 
class DynamicFilterMlp(nn.Module):    
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """    

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,   
                 bias=False, **kwargs):   
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)  
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()  
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)     
        self.drop2 = nn.Dropout(drop_probs[1])
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)   
        x = self.drop2(x)    
        return x

class DynamicFilter(nn.Module):
    def __init__(self, inc, dim, size=14, expansion_ratio=2, reweight_expansion_ratio=.25, 
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, weight_resize=False,   
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters     
        self.dim = dim   
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize    
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias) 
        self.act1 = act1_layer()     
        self.reweight = DynamicFilterMlp(dim, reweight_expansion_ratio, num_filters * self.med_channels) 
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,  
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()   
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)   
     
        self.conv1x1 = Conv(inc, dim, 1) if inc != dim else nn.Identity()  
     
    def forward(self, x):
        B, _, H, W, = x.shape

        x = self.conv1x1(x)    
        x = x.permute(0, 2, 3, 1)
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,   
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)    
        x = self.act1(x)  
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],     
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:  
            complex_weights = torch.view_as_complex(self.complex_weights) 
        routeing = routeing.to(torch.complex64)   
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)   
        if self.weight_resize:    
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)     
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)    
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')  

        x = self.act2(x)
        x = self.pwconv2(x)    
        return x.permute(0, 3, 1, 2)

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
 
    # 此模块不支持多尺度训练，且需要height=width，size参数可以填height或者width
    module = DynamicFilter(in_channel, out_channel, size=height).to(device)    

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET) 
