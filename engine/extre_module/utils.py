import os, contextlib, platform    
import matplotlib.pyplot as plt    
from pathlib import Path
from tqdm import tqdm as tqdm_original
     
RANK = int(os.getenv("RANK", -1))    
VERBOSE = str(os.getenv("DEIM_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format     
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

def emojis(string=""):   
    """Return platform-dependent emoji-safe version of string."""  
    return string.encode().decode("ascii", "ignore") if WINDOWS else string   

class TryExcept(contextlib.ContextDecorator):
    """ 
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.
 
    Examples:
        As a decorator:     
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():  
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """     

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings.""" 
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):    
        """Executes when entering TryExcept context, initializes instance."""    
        pass
   
    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))   
        return True

def plt_settings(rcparams=None, backend="Agg"):  
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function. 

    Example:   
        decorator: @plt_settings({"font.size": 12})  
        context manager: with plt_settings({"font.size": 12}):     

    Args:     
        rcparams (dict): Dictionary of rc parameters to set.  
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.    

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution. 
    """
    if rcparams is None:    
        rcparams = {"font.size": 11}  

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""
     
        def wrapper(*args, **kwargs): 
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8 
                plt.switch_backend(backend)    

            # Plot with backend and always revert to original backend
            try:    
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)     
            finally:
                if switch:   
                    plt.close("all")
                    plt.switch_backend(original_backend)    
            return result
  
        return wrapper 
 
    return decorator 

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.  

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a
    directory if it does not already exist.

    Args:     
        path (str | pathlib.Path): Path to increment.
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        sep (str): Separator to use between the path and the incrementation number.
        mkdir (bool): Create a directory if it does not exist.   
  
    Returns:  
        (pathlib.Path): Incremented path.
    
    Examples:
        Increment a directory path:
        >>> from pathlib import Path  
        >>> path = Path("runs/exp")    
        >>> new_path = increment_path(path)     
        >>> print(new_path)
        runs/exp2
  
        Increment a file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path) 
        >>> print(new_path)
        runs/exp/results2.txt    
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
    
        # Method 1     
        for n in range(2, 9999):  
            p = f"{path}{sep}{n}{suffix}"  # increment path 
            if not os.path.exists(p):
                break
        path = Path(p)  
  
    if mkdir:    
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

class TQDM(tqdm_original):    
    """     
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides   
    additional customization options.
  
    Attributes: 
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.     
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:   
        __init__: Initializes the TQDM object with custom settings.

    Examples:
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass     
    """    
  
    def __init__(self, *args, **kwargs):
        """
        Initializes a custom TQDM progress bar.     

        This class extends the original tqdm class to provide customized behavior for Ultralytics projects.  
  
        Args: 
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.     
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.

        Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.  
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.
     
        Examples: 
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):     
            ...     # Your code here
            ...     pass 
        """
        # kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)  # logical 'and' with default value if passed  
        # kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)