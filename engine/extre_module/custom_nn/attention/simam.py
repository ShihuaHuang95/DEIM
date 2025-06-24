'''  
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/ICML2021-SimAM.png
论文链接：https://proceedings.mlr.press/v139/yang21o/yang21o.pdf    
'''

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch     
import torch.nn as nn

class SimAM(torch.nn.Module):    
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda    
     
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s  

    @staticmethod 
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1 
     
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5 
  
        return x * self.activaton(y)
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = SimAM(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    # 这个模块完全没参数，测不了  
    # print(ORANGE)   
    # flops, macs, _ = calculate_flops(model=module,
    #                                  input_shape=(batch_size, channel, height, width),
    #                                  output_as_string=True,   
    #                                  output_precision=4,   
    #                                  print_detailed=True)
    # print(RESET)