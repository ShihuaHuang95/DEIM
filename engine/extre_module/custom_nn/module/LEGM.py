'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2024-LEGM.png
论文链接：https://arxiv.org/pdf/2403.01105   
'''
    
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   

import torch
import torch.nn as nn
import torch.nn.functional as F     
from timm.layers import trunc_normal_
  
from engine.extre_module.ultralytics_nn.conv import Conv 
 
# LEGM模块描述     
# 1. LEGM模块的应用场景与解决的问题
# Local Feature-Embedded Global Feature Extraction Module（LEGM）是一种高效的特征提取架构，专为需要兼顾局部细节与全局上下文的复杂视觉任务设计。该模块特别适用于目标检测、图像分类、语义分割以及视频分析等计算机视觉任务。LEGM通过其独特的多尺度特征融合与注意力机制，解决了传统模型在处理复杂场景时对局部特征与全局特征整合不足的问题，为高精度和高效能的特征表示提供了全新的解决方案。 
# 具体而言，LEGM能够有效应对以下挑战：

# 局部与全局特征的平衡：传统卷积神经网络或Transformer架构在捕捉全局上下文时可能忽略局部细节，而LEGM通过嵌入局部特征的全局特征提取机制，实现了两者的无缝融合。   
# 计算效率与性能的矛盾：LEGM结合了卷积操作与窗口化注意力机制，显著降低了计算复杂度，同时保持了强大的特征表达能力，适用于资源受限的实时应用场景。
# 尺度变化与背景干扰的鲁棒性：通过动态窗口移位和多头注意力设计，LEGM增强了模型对多尺度目标和复杂背景的适应能力，特别适合处理高分辨率或动态场景的视觉任务。   
   
# 2. LEGM模块的创新点与优点
# LEGM模块在设计上融合了多种前沿理念，展现出显著的创新性与工程价值。其创新点和优点包括以下几个方面： 
# 创新点:  
   
# 局部特征嵌入的全局注意力机制LEGM通过WATT（Window-based Attention with Relative Position Bias）模块，将局部特征的卷积处理与全局特征的注意力机制深度结合。这种局部嵌入式全局提取策略，不仅增强了特征的语义丰富性，还显著提升了模型对复杂场景的理解能力。    

# 动态窗口移位与混合卷积策略模块创新性地引入了动态窗口移位机制（shift_size），配合多种卷积类型（如深度可分离卷积和标准卷积），实现了灵活的特征捕捉方式。这种设计有效扩展了感受野，同时降低了计算开销。

# 自适应层归一化（LayNormal）设计LEGM引入了LayNormal模块，通过学习输入特征的均值和方差动态调整归一化参数，增强了模型对不同数据分布的适应性。这种自适应归一化机制为特征处理提供了更高的灵活性和鲁棒性。
 

# 优点:

# 高效的多尺度特征融合相较于传统的Transformer或CNN模块，LEGM在保持全局特征提取能力的同时，通过局部特征的嵌入式设计，显著提升了多尺度目标的检测精度，尤其适用于高分辨率图像处理。 
  
# 模块化与灵活性LEGM采用高度模块化的设计，参数（如窗口大小、头数、卷积类型等）可根据任务需求灵活调整，使其易于集成到多种主流架构（如ResNet、YOLO或Swin Transformer）中，具有广泛的适用性。
    
# 轻量化和高性能的协同优化通过结合深度可分离卷积和窗口化注意力机制，LEGM在大幅降低计算量和参数量的同时，依然保持了卓越的性能表现，非常适合边缘设备或实时推理场景。 
 
# 综上所述，LEGM模块以其创新的局部嵌入式全局特征提取机制、动态窗口化注意力设计以及自适应归一化策略，为计算机视觉领域提供了一种高效、灵活且鲁棒的特征提取解决方案。其独特的设计理念不仅推动了高性能视觉模型的发展，也为学术研究与工业应用开辟了新的可能性。     
    

    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)    
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
    return windows
     

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)     
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)     
    return x  

def get_relative_positions(window_size): 
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    # coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='xy'))   
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()   
    relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())    
    return relative_positions_log
 
class WATT(nn.Module): 
    def __init__(self, dim, window_size, num_heads): 
  
        super().__init__()    
        self.dim = dim    
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)  
        self.register_buffer("relative_positions", relative_positions)   
        self.meta = nn.Sequential( 
            nn.Linear(2, 256, bias=True),     
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)     
        )
    
        self.softmax = nn.Softmax(dim=-1)
  
    def forward(self, qkv):
        B_, N, _ = qkv.shape    
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale 
        attn = (q @ k.transpose(-2, -1))     
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x   
  
class Att(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()     
        self.dim = dim
        self.head_dim = int(dim // num_heads) 
        self.num_heads = num_heads
        self.window_size = window_size    
        self.shift_size = shift_size    
        self.use_attn = use_attn   
        self.conv_type = conv_type    

        if self.conv_type == 'Conv':    
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect') 
            ) 

        if self.conv_type == 'DWConv':     
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect') 
        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)   
        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WATT(dim, window_size, num_heads) 
 
    def check_size(self, x, shift=False): 
        _, _, h, w = x.size()  
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')    
        else: 
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')   
        return x    
  
    def forward(self, X):
        B, C, H, W = X.shape
    
        if self.conv_type == 'DWConv' or self.use_attn:  
            V = self.V(X)   

        if self.use_attn:  
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)     
 
            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0) 
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)    
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C     
   
            attn_windows = self.attn(qkv)
 
            # merge windows   
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift  
            out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            attn_out = out.permute(0, 3, 1, 2)
     
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else: 
                out = self.proj(attn_out) 

        else:
            if self.conv_type == 'Conv': 
                out = self.conv(X)
            elif self.conv_type == 'DWConv':     
                out = self.proj(self.conv(V))
  
        return out   
    
class Mlp(nn.Module):  
    def __init__(self, in_features, hidden_features=None, out_features=None):  
        super().__init__()   
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(   
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )
    
    def forward(self, x):  
        return self.mlp(x)

class LayNormal(nn.Module):  
    def __init__(self, dim, eps=1e-5, detach_grad=False): 
        super(LayNormal, self).__init__()     
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=.02)    
        nn.init.constant_(self.meta1.bias, 1)   
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)     
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)     
        normalized_input = (input - mean) / std
        if self.detach_grad:   
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean) 
        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias
  
class LEGM(nn.Module):
    def __init__(self, inc, dim, num_heads=8, mlp_ratio=4.,
                 norm_layer=LayNormal, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn    
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()   
        self.attn = Att(dim, num_heads=num_heads, window_size=window_size,  
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type) 
     
        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity() 
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))    

        self.conv1x1 = Conv(inc, dim, 1) if inc != dim else nn.Identity()
   
    def forward(self, x):
        x = self.conv1x1(x)  

        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)     
        x = self.attn(x)    
        if self.use_attn: x = x * rescale + rebias 
        x = identity + x   

        identity = x  
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias  
        x = identity + x   
 
        return x

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    module = LEGM(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)  
