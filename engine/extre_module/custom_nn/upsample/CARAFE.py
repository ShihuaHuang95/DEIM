'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2019-CARAFE.png   
论文链接：https://arxiv.org/abs/1905.02188  
'''
 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
     
import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops
  
import torch
import torch.nn as nn     
   
from engine.extre_module.ultralytics_nn.conv import Conv     

class CARAFE(nn.Module):   
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2): 
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188". 
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.  
            scale: The expected upsample scale.  
            k_up: The size of the reassembly kernel. 
            k_enc: The kernel size of the encoder.
        Returns:  
            X: The upsampled feature map. 
        """
        super(CARAFE, self).__init__()
        self.scale = scale    

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)     
 
    def forward(self, X):  
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_  
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_   

        X = self.upsmp(X)                               # b * c * h_ * w_  
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_
  
        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X   
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = CARAFE(channel).to(device)    

    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET) 
