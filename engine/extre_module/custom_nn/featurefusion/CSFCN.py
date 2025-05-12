'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IEEETIP2023-CSFCN.png
论文链接：https://ieeexplore.ieee.x-lib.xyz/document/10268334  
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
    
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)     
    # (1, 4, 8,12)     
    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()    

        self.grids = grids
        self.channels = channels

    def forward(self, feats):

        b, c , h , w = feats.size()
        ar = w / h 

        return torch.cat([  
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)
    
class LocalAttenModule(nn.Module):    
    def __init__(self, in_channels=256, inter_channels=32):
        super(LocalAttenModule, self).__init__()    

        self.conv = nn.Sequential(   
            Conv(in_channels, inter_channels,1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False)) 

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()     
    def keras_init_weight(self):  
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu')) 
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)    

    def forward(self, x):     
        res1 = x 
        res2 = x   

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)
    
        res1 = res1 * x_mask    
  
        return res1 + res2

class CFC_CRB(nn.Module): 
    def __init__(self, in_channels=512, grids=(6, 3, 2, 1)): # 先ce后ffm  

        super(CFC_CRB, self).__init__()  
        self.grids = grids  
        inter_channels = in_channels    
        self.inter_channels = inter_channels
     
        self.reduce_channel = Conv(in_channels, inter_channels, 3)  
        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1) 
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)     
        self.key_channels = 32   

        self.value_psp = PSPModule(grids, inter_channels)   
        self.key_psp = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)    

        self.local_attention = LocalAttenModule(inter_channels, inter_channels//8)
        self.keras_init_weight()
        
    def keras_init_weight(self):
        for ly in self.children(): 
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)  
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))  
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):

        x = self.reduce_channel(x) # 降维- 128

        m_batchsize,_,h,w = x.size() 
     
        query = self.query_conv(x).view(m_batchsize,32,-1).permute(0,2,1) ##  b c n ->  b n c  

        key = self.key_conv(self.key_psp(x))  ## b c s    

        sim_map = torch.matmul(query,key)

        sim_map = self.softmax(sim_map)
        # sim_map = self.attn_drop(sim_map)
        value = self.value_conv(self.value_psp(x)) #.permute(0,2,1)  ## b c s 

        # context = torch.matmul(sim_map,value) ## B N S * B S C ->  B N C 
        context = torch.bmm(value,sim_map.permute(0,2,1))  #  B C S * B S N - >  B C N

        # context = context.permute(0,2,1).view(m_batchsize,self.inter_channels,h,w) 
        context = context.view(m_batchsize,self.inter_channels,h,w)
        # out = x + self.gamma * context     
        context = self.local_attention(context)
   
        out = x + context

        return out
    
class SFC_G2(nn.Module):
    def __init__(self, inc, ouc):
        super(SFC_G2, self).__init__()  
     
        self.groups = 2
        self.conv_8 = Conv(inc[1], ouc, 3)     
        self.conv_32 = Conv(inc[0], ouc, 3)   

        self.conv_offset = nn.Sequential(     
            Conv(ouc * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )
     
        self.keras_init_weight()
        self.conv_offset[1].weight.data.zero_()
     
    def keras_init_weight(self):   
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)     
                
    def forward(self, x):   
        sp, cp = x
        n, _, out_h, out_w = cp.size()   
    
        # x_32
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)
        # x_8
        cp = self.conv_8(cp)
   
        conv_results = self.conv_offset(torch.cat([cp, sp], 1))
     
        sp = sp.reshape(n*self.groups,-1,out_h,out_w)
        cp = cp.reshape(n*self.groups,-1,out_h,out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)
  
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)  
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)   
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)     
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)    
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)    

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm   
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm     
  
        cp = F.grid_sample(cp, grid_l , align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h , align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)    
        sp = sp.reshape(n, -1, out_h, out_w)     

        att = 1 + torch.tanh(conv_results[:, self.groups*4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp

class CSFCN(nn.Module):
    def __init__(self, inc, ouc) -> None: 
        super().__init__()    

        self.CFC_CRB = CFC_CRB(inc[0])
        self.SFC_G2 = SFC_G2(inc, ouc)  
    
    def forward(self, x):
        p3, p5 = x     
        p3 = self.CFC_CRB(p3)
        p5 = self.SFC_G2((p3, p5))
        return p3, p5

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel_p3, height_p3, width_p3 = 1, 32, 80, 80
    batch_size, channel_p5, height_p5, width_p5 = 1, 64, 20, 20
    ouc_channel = 32   
    inputs_1 = torch.randn((batch_size, channel_p3, height_p3, width_p3)).to(device) 
    inputs_2 = torch.randn((batch_size, channel_p5, height_p5, width_p5)).to(device)  
     
    module = CSFCN([channel_p3, channel_p5], ouc_channel).to(device)  

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'p3.size:{inputs_1.size()} p5.size:{inputs_2.size()} outputs_p3.size:{outputs[0].size()} outputs_p5.size:{outputs[1].size()}' + RESET)   
    
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)     
    print(RESET)