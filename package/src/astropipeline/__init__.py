import sys
import astropy.io.fits as pyfits

# Store the original open function to avoid recursion
_orig_open = pyfits.open

def _custom_open(name, *args, **kwargs):
    # If on Windows and memmap is not explicitly set, default to False
    if sys.platform.startswith('win'):
        if 'memmap' not in kwargs:
            kwargs['memmap'] = False
    return _orig_open(name, *args, **kwargs)

# Override in the module to prevent Windows file locking issues in notebook environments
pyfits.open = _custom_open
