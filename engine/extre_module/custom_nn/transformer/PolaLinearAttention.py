'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/ICLR2025-PolaLinearAttention.png    
论文链接：https://arxiv.org/abs/2501.15061  
'''
    
import warnings  
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
   
import torch    
import torch.nn as nn
   
class PolaLinearAttention(nn.Module):    
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."    

        self.h = hw[0]    
        self.w = hw[1]
 
        self.dim = dim 
        self.num_heads = num_heads
        head_dim = dim // num_heads     
        self.head_dim = head_dim     

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias) 
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)   
        self.proj_drop = nn.Dropout(proj_drop)
  
        self.sr_ratio = sr_ratio  
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim) 
 
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2) 
     
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha  

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim))) 
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))
 
    def forward(self, x):     
        B, N, C = x.shape  
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)
  
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)  
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) 
            x_ = self.norm(x_)  
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:    
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3) 
        k, v = kv[0], kv[1]  
        n = k.shape[1]  

        k = k + self.positional_encoding    
        kernel_function = nn.ReLU() 
        
        scale = nn.Softplus()(self.scale)  
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)
        
        q = q / scale    
        k = k / scale    
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() 
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()     
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()     
   
        q_pos = kernel_function(q) ** power   
        q_neg = kernel_function(-q) ** power 
        k_pos = kernel_function(k) ** power 
        k_neg = kernel_function(-k) ** power 

        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)   
        k = torch.cat([k_pos, k_neg],dim=-1)   
    
        v1,v2 = torch.chunk(v,2,dim=-1)   
 
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z   
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))   
        x_opp = q_opp @ kv * z
     
        x = torch.cat([x_sim, x_opp],dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1: 
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)    
 
        v = v.reshape(B * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v  
        x = x * g     

        x = self.proj(x)  
        x = self.proj_drop(x)
  
        return x     
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 32, 32     
    inputs = torch.randn((batch_size, height * width, channel)).to(device)    
  
    # 此模块不支持多尺度训练，hw的参数是一个元组，其为当前特征图的height,width。
    module = PolaLinearAttention(channel, hw=(height, width)).to(device)
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
   
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, height * width, channel),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)