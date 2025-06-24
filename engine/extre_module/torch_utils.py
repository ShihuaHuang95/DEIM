import torch, torchvision
import torch.nn as nn
   
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

def check_cuda():   
    print(GREEN + f"PyTorch 版本: {torch.__version__}")   
    print(f"Torchvision 版本: {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if cuda_available:  
        device_count = torch.cuda.device_count()    
        print(f"GPU 数量: {device_count}")
        
        for i in range(device_count):     
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB") 
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}") 
        
        print(f"当前设备索引: {torch.cuda.current_device()}")   
        print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}" + RESET)  
  
def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(     
            conv.in_channels,     
            conv.out_channels,
            kernel_size=conv.kernel_size,   
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,    
        )     
        .requires_grad_(False)
        .to(conv.weight.device)     
    )     
   
    # Prepare filters   
    w_conv = conv.weight.view(conv.out_channels, -1)     
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))  

    # Prepare spatial bias    
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv    

def model_fuse_test(model):
    model.eval()     
    for name, m in model.named_modules():  
        if hasattr(m, 'convert_to_deploy'): 
            print(BLUE + f"Converting module: {m.__class__}" + RESET)
            m.convert_to_deploy()
    return model  
