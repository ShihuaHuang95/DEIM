'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2024-CGLU.png
论文链接：https://arxiv.org/pdf/2311.17132   
'''    
 
import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops   
     
import torch 
import torch.nn as nn

class ConvolutionalGLU(nn.Module):
    """
    ConvolutionalGLU（卷积门控线性单元）模块
    
    该模块结合了通道分割、深度可分离卷积和门控机制，以提高特征表达能力。
    
    参数：    
        in_features (int): 输入通道数。     
        hidden_features (int, 可选): 隐藏层通道数，默认为输入通道数。 
        out_features (int, 可选): 输出通道数，默认为输入通道数。
        act_layer (nn.Module, 可选): 激活函数，默认使用 GELU。     
        drop (float, 可选): Dropout 概率，默认值为 0。
    """     
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:  
        super().__init__()
        
        # 如果未指定 out_features 和 hidden_features，则默认与 in_features 一致
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features     
        
        # 计算隐藏层通道数，并保证为 2/3 的比例
        hidden_features = int(2 * hidden_features / 3) 
    
        # 1x1 卷积用于通道扩展，并进行通道分割（GLU 机制）
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)     
   
        # 深度可分离卷积层，提取局部特征
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),    
            act_layer()  # 激活函数（默认使用 GELU）
        )
  
        # 1x1 卷积用于恢复通道数 
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  
        
        # Dropout 层，防止过拟合
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        前向传播过程：  
        1. 先存储输入 x 作为残差连接的 shortcut。
        2. 通过 1x1 卷积 self.fc1，将输入通道扩展为 2 倍，并分成两个部分 (x, v)。
        3. x 经过深度可分离卷积 self.dwconv 处理后，与门控信号 v 相乘，实现门控机制。
        4. 经过 Dropout 防止过拟合。
        5. 通过 1x1 卷积 self.fc2 将通道数恢复到输出通道数。
        6. 再次进行 Dropout。
        7. 残差连接，将原始输入 x_shortcut 与处理后的 x 相加。   
        """
        
        # 残差连接的快捷分支
        x_shortcut = x    
     
        # 1x1 卷积，通道扩展并分割为 x 和门控信号 v
        x, v = self.fc1(x).chunk(2, dim=1)    
     
        # 深度可分离卷积处理，并通过门控信号 v 进行调制    
        x = self.dwconv(x) * v
        
        # Dropout 以减少过拟合  
        x = self.drop(x)  
        
        # 通过 1x1 卷积恢复通道数
        x = self.fc2(x)   
        
        # 再次进行 Dropout 
        x = self.drop(x)    
    
        # 残差连接，最终输出     
        return x_shortcut + x  
   
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
   
    module = ConvolutionalGLU(in_features=in_channel, out_features=out_channel).to(device) 
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)  
