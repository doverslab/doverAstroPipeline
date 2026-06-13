import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    # Attempt to query device count to ensure CUDA drivers and devices are functional
    if cp.cuda.runtime.getDeviceCount() > 0:
        HAS_GPU = True
    else:
        HAS_GPU = False
except Exception:
    cp = None
    cp_ndimage = None
    HAS_GPU = False

def to_gpu(arr):
    """Move an array to GPU if a GPU is available, otherwise return as-is."""
    if HAS_GPU and arr is not None:
        if cp is not None and isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move a GPU array to CPU, otherwise return as-is."""
    if HAS_GPU and cp is not None:
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return arr

def get_array_module(arr):
    """Return cupy if array is on GPU, otherwise numpy."""
    if HAS_GPU and cp is not None:
        return cp.get_array_module(arr)
    return np

def get_ndimage():
    """Return cupyx.scipy.ndimage if GPU is available, otherwise scipy.ndimage."""
    if HAS_GPU and cp_ndimage is not None:
        return cp_ndimage
    import scipy.ndimage as ndimage
    return ndimage
