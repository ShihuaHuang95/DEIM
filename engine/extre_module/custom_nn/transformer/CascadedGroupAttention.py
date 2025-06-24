'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2023-Cascaded Group Attention.png
论文链接：https://arxiv.org/pdf/2305.07027
'''   

import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops    

import itertools
import torch
    
from engine.extre_module.ultralytics_nn.conv import Conv

class CascadedGroupAtt(torch.nn.Module):
    r""" Cascaded Group Attention.   

    Args: 
        dim (int): Number of input channels.  
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.     
        resolution (int): Input resolution, correspond to the window size.    
        kernels (List[int]): The kernel size of the dw conv on query.
    """     
    def __init__(self, dim, key_dim, num_heads=4, 
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5]):
        super().__init__()   
        self.num_heads = num_heads    
        self.scale = key_dim ** -0.5    
        self.key_dim = key_dim    
        self.d = dim // num_heads     
        self.attn_ratio = attn_ratio 
   
        qkvs = []    
        dws = []  
        for i in range(num_heads):
            qkvs.append(Conv(dim // (num_heads), self.key_dim * 2 + self.d, act=False))
            dws.append(Conv(self.key_dim, self.key_dim, kernels[i], g=self.key_dim, act=False))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv(self.d * num_heads, dim, act=False))
 
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)   
        attention_offsets = {}
        idxs = []
        for p1 in points: 
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))   
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()     
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'): 
            del self.ab     
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)  
        B, C, H, W = x.shape     
        trainingab = self.attention_biases[:, self.attention_bias_idxs]  
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input     
                feat = feat + feats_in[i]
            feat = qkv(feat)   
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q) 
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N 
            attn = (     
                (q.transpose(-2, -1) @ k) * self.scale    
                + 
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))   
        return x
   

class CascadedGroupAttention(torch.nn.Module):
    r""" Local Window Attention. CVPR2023-EfficientViT 

    Args:    
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.   
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.    
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim=16, num_heads=4, 
                 attn_ratio=4,
                 resolution=14,    
                 window_resolution=7,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution    
        assert window_resolution > 0, 'window_size must be greater than 0' 
        self.window_resolution = window_resolution
 
        self.attn = CascadedGroupAtt(dim, key_dim, num_heads, 
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution, 
                                kernels=kernels) 
   
    def forward(self, x):    
        B, C, H, W = x.shape
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %  
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %   
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b)) 

            pH, pW = H + pad_b, W + pad_r  
            nH = pH // self.window_resolution   
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C 
            ).permute(0, 3, 1, 2) 
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC  
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,     
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:    
                x = x[:, :H, :W].contiguous() 

            x = x.permute(0, 3, 1, 2) 
 
        return x   
 
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = CascadedGroupAttention(channel).to(device)
   
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
    
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width), 
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)  
