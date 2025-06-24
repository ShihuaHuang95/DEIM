import contextlib, time   
import torch 

class Profile(contextlib.ContextDecorator):  
    """  
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from ultralytics.utils.ops import Profile    

        with Profile(device=device) as dt:     
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"     
        ```  
    """     

    def __init__(self, t=0.0, device: torch.device = None):    
        """     
        Initialize the Profile class.     

        Args:     
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu). 
        """ 
        self.t = t   
        self.device = device 
        self.cuda = bool(device and str(device).startswith("cuda"))
   
    def __enter__(self):  
        """Start timing."""    
        self.start = self.time()     
        return self 
    
    def __exit__(self, type, value, traceback):  # noqa 
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time  
        self.t += self.dt  # accumulate dt  

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"  

    def time(self):
        """Get current time."""     
        if self.cuda: 
            torch.cuda.synchronize(self.device)
        return time.time()  
