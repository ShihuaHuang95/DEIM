''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2023-iRMB.png     
论文链接：https://arxiv.org/abs/2301.01146    
''' 
   
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch  
import torch.nn as nn 
import torch.nn.functional as F  
import torch.nn.init as init 
from timm.layers import DropPath 
from einops import rearrange  
     
from engine.extre_module.ultralytics_nn.conv import Conv

class SEAttention(nn.Module):     
    def __init__(self, channel=512,reduction=16):    
        super().__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),   
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )  
 
    def init_weights(self):   
        for m in self.modules():    
            if isinstance(m, nn.Conv2d):   
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)    
            elif isinstance(m, nn.BatchNorm2d): 
                init.constant_(m.weight, 1)     
                init.constant_(m.bias, 0)     
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:    
                    init.constant_(m.bias, 0)    

    def forward(self, x):   
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)     
        return x * y.expand_as(x)

class iRMB(nn.Module):   
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,     
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__() 
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()  
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip  
		self.attn_s = attn_s
		if self.attn_s:     
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads' 
			self.dim_head = dim_head  
			self.window_size = window_size   
			self.num_head = dim_in // dim_head   
			self.scale = self.dim_head ** -0.5 
			self.attn_pre = attn_pre
			self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
			self.v = nn.Sequential(    
				nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
				self.act
			)     
			self.attn_drop = nn.Dropout(attn_drop)  
		else:
			if v_proj:     
				self.v = nn.Sequential(    
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act 
				)
			else:   
				self.v = nn.Identity()   
		self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)    
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()    
  
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding   
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H     
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0     
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H    
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,)) 
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W    
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()   
			# attention   
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()     
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1) 
			attn_spa = self.attn_drop(attn_spa)  
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:    
				v = self.v(x)  
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v    
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:  
			x = self.v(x)
  
		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
    
		x = self.proj_drop(x)    
		x = self.proj(x)    
   
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
  
    module = iRMB(in_channel, out_channel).to(device) 
  
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)