'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICLR2022-DPBAttention.png
论文链接：https://arxiv.org/pdf/2108.00154    
'''   

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch 
import torch.nn as nn   

class DynamicPosBias(nn.Module):  
    r"""DPB module   
    
    Use a MLP to predict position bias used in attention.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4     
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(   
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),   
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)  
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),    
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)   
        )     
    def forward(self, biases):
        if self.residual: 
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)     
            pos = pos + self.pos2(pos)   
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos     
   
    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads     
        return flops

class DPB_Attention(nn.Module):     
    r""" Multi-head self attention module with relative position bias. 
    Args:  
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. 
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set    
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0    
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0 
    """ 
  
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,   
                 position_bias=True):

        super().__init__()   
        self.dim = dim
        self.num_heads = num_heads   
        head_dim = dim // num_heads   
        self.scale = qk_scale or head_dim ** -0.5   
        self.position_bias = position_bias
        if self.position_bias:     
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
    
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x, mask=None):   
        """   
        Args:
            x: input features with shape of (num_windows*B, N, C) 
            mask: (0/-inf) mask with shape of (num_windows, Gh*Gw, Gh*Gw) or None
        """
        B_, C, H, W = x.shape
        group_size, N = (H, W), H * W     
        x = x.flatten(2).permute(0, 2, 1)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
     
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous()) # (num_windows*B, N, N), N = Gh*Gw

        if self.position_bias:
            # generate mother-set     
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)     
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float() 

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw  
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2     
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0  
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw     

            pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads
            # select position bias   
            relative_position_bias = pos[relative_position_index.view(-1)].view(     
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw  
            attn = attn + relative_position_bias.unsqueeze(0) 

        if mask is not None:    
            nG = mask.shape[0]   
            attn = attn.view(B_ // nG, nG, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # (B, nG, nHead, N, N)     
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)  
        else:     
            attn = self.softmax(attn)  

        attn = self.attn_drop(attn)     
   
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)     
        x = self.proj_drop(x)
        return x.permute(0, 2, 1).view([B_, C, H, W]).contiguous()    
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 32, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  

    module = DPB_Attention(channel, num_heads=8).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
    
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)     
    print(RESET)     
