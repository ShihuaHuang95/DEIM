'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/WACV2024-DeformableLKA.png
论文链接：https://arxiv.org/abs/2309.00121
'''
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch, torchvision     
import torch.nn as nn    

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):  
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,   
                                    dilation=dilation,
                                    bias=True)  

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels, 
                                                        kernel_size=kernel_size,   
                                                        padding=padding,
                                                        groups=groups,    
                                                        stride=stride,     
                                                        dilation=dilation,
                                                        bias=False)     

    def forward(self, x):
        offsets = self.offset_net(x) 
        out = self.deform_conv(x, offsets)
        return out    

class DeformableLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)     
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)    
   
    def forward(self, x):    
        u = x.clone()   
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)    
        attn = self.conv1(attn)
        return u * attn  

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 32, 32   
    inputs = torch.randn((batch_size, channel, height, width)).to(device) 
    
    module = DeformableLKA(channel).to(device)     

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET) 
