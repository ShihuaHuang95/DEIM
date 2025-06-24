'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IJCAI2024-DFFN.png
论文链接：https://www.ijcai.org/proceedings/2024/0081.pdf
'''

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch
import torch.nn as nn    
   
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
     
class FrequencyGate(nn.Module):
    """ Frequency-Gate. 
    Args:   
        dim (int): Input channels.   
    """   
    def __init__(self, dim):
        super().__init__()     
        self.norm = LayerNormGeneral((dim, 1, 1), normalized_dim=(1, 2, 3))
        self.conv = nn.Sequential( 
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        )   

    def forward(self, x): 
        x1, x2 = x.chunk(2, dim =1)
        x2 = self.conv(self.norm(x2))  
        return x1 * x2

class DFFN(nn.Module):
    """ Dual frequency aggregation Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.  
        hidden_features (int | None): Number of hidden channels. Default: None 
        out_features (int | None): Number of output channels. Default: None    
        act_layer (nn.Module): Activation layer. Default: nn.GELU 
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):    
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features     
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  
        self.act = act_layer()     
        self.fg = FrequencyGate(hidden_features//2)
        self.fc2 = nn.Conv2d(hidden_features//2, out_features, 1)  
        self.drop = nn.Dropout(drop)   
 
    def forward(self, x):     
        x = self.fc1(x)    
        x = self.act(x)
        x = self.drop(x)
        x = self.fg(x)
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x)  
        return x
   
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
     
    module = DFFN(in_features=in_channel, out_features=out_channel).to(device) 
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)    
