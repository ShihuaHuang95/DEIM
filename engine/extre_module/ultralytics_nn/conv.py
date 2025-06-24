# Ultralytics YOLO 🚀, AGPL-3.0 license     
"""Convolution modules."""
     
import math   
     
import numpy as np     
import torch    
import torch.nn as nn 
from ..torch_utils import fuse_conv_and_bn

__all__ = (  
    "Conv",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d", 
    "ConvTranspose",    
    "GhostConv",    
    "RepConv",
    "DSConv"
)    

   
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size   
    if p is None: 
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad    
    return p  

     
class Conv(nn.Module):    
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation  

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)   
        self.bn = nn.BatchNorm2d(c2)     
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity() 

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):   
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))  
    
    def convert_to_deploy(self): 
        if hasattr(self, "bn"):
            self.conv = fuse_conv_and_bn(self.conv, self.bn)  # update conv    
            delattr(self, "bn")  # remove batchnorm 
            self.forward = self.forward_fuse  # update forward   
  
class LightConv(nn.Module):
    """  
    Light convolution with args(ch_in, ch_out, kernel).
    
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py 
    """   
    
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation.""" 
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)   
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))
 
class DWConv(Conv):  
    """Depth-wise convolution."""
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters.""" 
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DSConv(nn.Module): 
    """Depthwise Separable Convolution"""   
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True) -> None:
        super().__init__()  
        
        self.dwconv = DWConv(c1, c1, 3)    
        self.pwconv = Conv(c1, c2, 1)    
    
    def forward(self, x):
        return self.pwconv(self.dwconv(x))

class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""     
  
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out     
        """Initialize DWConvTranspose2d class with given parameters.""" 
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))     
 

class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer.""" 

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()    
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()  
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):     
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))    

class GhostConv(nn.Module):    
    """Ghost Convolution https://github.com/huawei-noah/ghostnet.""" 
    
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)   
   
    def forward(self, x):   
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""   
        y = self.cv1(x)   
        return torch.cat((y, self.cv2(y)), 1)     

     
class RepConv(nn.Module):     
    """  
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """     
   
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):   
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()     
        assert k == 3 and p == 1    
        self.g = g
        self.c1 = c1
        self.c2 = c2     
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None  
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)   
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):     
        """Forward process."""  
        return self.act(self.conv(x))

    def forward(self, x):     
        """Forward process."""  
        id_out = 0 if self.bn is None else self.bn(x)   
        return self.act(self.conv1(x) + self.conv2(x) + id_out)  

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""    
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)   
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)  
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
 
    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):  
        """Pads a 1x1 tensor to a 3x3 tensor."""     
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""  
        if branch is None:   
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight  
            beta = branch.bn.bias    
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):     
            if not hasattr(self, "id_tensor"):  
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1    
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)  
            kernel = self.id_tensor 
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight    
            beta = branch.bias    
            eps = branch.eps    
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)   
        return kernel * t, beta - running_mean * gamma / std
     
    def convert_to_deploy(self):  
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""     
        if hasattr(self, "conv"):    
            return
        kernel, bias = self.get_equivalent_kernel_bias() 
        self.conv = nn.Conv2d(     
            in_channels=self.conv1.conv.in_channels,    
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,     
            padding=self.conv1.conv.padding,   
            dilation=self.conv1.conv.dilation,  
            groups=self.conv1.conv.groups,    
            bias=True,
        ).requires_grad_(False)   
        self.conv.weight.data = kernel
        self.conv.bias.data = bias    
        for para in self.parameters():   
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):   
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):  
            self.__delattr__("id_tensor") 
        self.forward = self.forward_fuse