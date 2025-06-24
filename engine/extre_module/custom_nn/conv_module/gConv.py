'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/gConv.png 
论文链接：https://arxiv.org/abs/2209.11448     
'''
   
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch, math     
import torch.nn as nn     
from torch.nn.init import trunc_normal_
   
from engine.extre_module.ultralytics_nn.conv import Conv  
     
class gConv(nn.Module): 
	def __init__(self, in_dim, dim, kernel_size=3, gate_act=nn.Sigmoid):   
		super().__init__()     
		self.dim = dim

		self.kernel_size = kernel_size

		self.norm_layer = nn.BatchNorm2d(dim)   
        
		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),   
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')  
		)  
     
		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)
    
		self.proj = nn.Conv2d(dim, dim, 1) 
     
		self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()

	def forward(self, X):     
		X = self.conv1x1(X)    
		iden = X     
		X = self.norm_layer(X)   
		out = self.Wv(X) * self.Wg(X)     
		out = self.proj(out)   
		return out + iden 

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
  
    module = gConv(in_channel, out_channel).to(device)
   
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)