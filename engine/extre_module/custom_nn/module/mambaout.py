''' 
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/CVPR2025-MambaOut.png
engine/extre_module/module_images/CVPR2025-MambaOut-UniRepBlock.png
engine/extre_module/module_images/CVPR2025-MambaOut-DRC.png  
论文链接：https://arxiv.org/abs/2405.07992
论文链接：https://arxiv.org/abs/2311.15599
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops   

import torch
import torch.nn as nn     
from functools import partial   
from timm.layers import DropPath 

from engine.extre_module.custom_nn.conv_module.DilatedReparamConv import DilatedReparamConv
from engine.extre_module.custom_nn.module.UniRepLKBlock import UniRepLKNetBlock
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.torch_utils import model_fuse_test   
 
class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.    

    Args:     
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm, 
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.     
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),  
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),   
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.   
    
        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):  
            For input shape of (B, N, C),    
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),     
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True. 

        For the several metaformer baslines,  
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);  
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).  
    """    
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True,     
        bias=True, eps=1e-5):     
        super().__init__()
        self.normalized_dim = normalized_dim   
        self.use_scale = scale
        self.use_bias = bias    
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None    
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps
     
    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale: 
            x = x * self.weight   
        if self.use_bias:
            x = x + self.bias
        return x     
     
class MambaOut(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.     
            Conduct convolution on partial channels can improve practical efficiency.    
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)     
    """
    def __init__(self, inc, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,  
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)),    
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()     
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)     
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)   
        self.act = act_layer()     
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv1x1 = Conv(inc, dim, 1) if inc != dim else nn.Identity()

    def forward(self, x):
        x = self.conv1x1(x)
        shortcut = x # [B, H, W, C] 
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1) 
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W] 
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]  
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x) 
        return x + shortcut   
    
class MambaOut_DilatedReparamConv(MambaOut):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667) 
    """   
    def __init__(self, inc, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1, norm_layer=partial(LayerNormGeneral, eps=0.000001, normalized_dim=(1, 2, 3)), act_layer=nn.GELU, drop_path=0, **kwargs):    
        super().__init__(inc, dim, expansion_ratio, kernel_size, conv_ratio, norm_layer, act_layer, drop_path, **kwargs) 
        conv_channels = int(conv_ratio * dim)
        self.conv = DilatedReparamConv(conv_channels, conv_channels, kernel_size=kernel_size)
     
class MambaOut_UniRepLKBlock(MambaOut):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.     
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and   
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """    
    def __init__(self, inc, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1, norm_layer=partial(LayerNormGeneral, eps=0.000001, normalized_dim=(1, 2, 3)), act_layer=nn.GELU, drop_path=0, **kwargs):
        super().__init__(inc, dim, expansion_ratio, kernel_size, conv_ratio, norm_layer, act_layer, drop_path, **kwargs)
        conv_channels = int(conv_ratio * dim)
        self.conv = UniRepLKNetBlock(conv_channels, conv_channels, kernel_size=kernel_size)   
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    print(RED + '-'*20 + " MambaOut " + '-'*20 + RESET)  

    module = MambaOut(in_channel, out_channel).to(device)  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
    
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)    
    
    print(RED + '-'*20 + " MambaOut_DilatedReparamConv " + '-'*20 + RESET)    

    module = MambaOut_DilatedReparamConv(in_channel, out_channel, kernel_size=11).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
   
    print(GREEN + 'test reparameterization.' + RESET) 
    module = model_fuse_test(module)
    outputs = module(inputs) 
    print(GREEN + 'test reparameterization done.' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)  
    
    print(RED + '-'*20 + " MambaOut_UniRepLKBlock " + '-'*20 + RESET)
    
    module = MambaOut_UniRepLKBlock(in_channel, out_channel, kernel_size=11).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     

    print(GREEN + 'test reparameterization.' + RESET) 
    module = model_fuse_test(module) 
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET) 

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True,   
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)   
