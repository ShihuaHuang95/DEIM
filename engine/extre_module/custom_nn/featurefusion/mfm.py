'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2024-MFM.png 
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
  
from engine.extre_module.ultralytics_nn.conv import Conv 

class MFM(nn.Module):
    def __init__(self, inc, dim, reduction=8):
        super(MFM, self).__init__()   

        self.height = len(inc)
        d = max(int(dim/reduction), 4)
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(    
            nn.Conv2d(dim, d, 1, bias=False),     
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:    
                self.conv1x1.append(Conv(i, dim, 1))
            else:    
                self.conv1x1.append(nn.Identity())     

    def forward(self, in_feats_):  
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))   

        B, C, H, W = in_feats[0].shape  

        in_feats = torch.cat(in_feats, dim=1)   
        in_feats = in_feats.view(B, self.height, C, H, W)    

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))     
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))  
     
        out = torch.sum(in_feats*attn, dim=1)
        return out

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32  
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)  
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)    

    module = MFM([channel_1, channel_2], ouc_channel).to(device)
     
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,     
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)