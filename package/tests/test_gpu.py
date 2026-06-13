import numpy as np
import pytest
from package.src.astropipeline import astropipeline_gpu as apgpu

def test_gpu_detection_fallback():
    # Verify target attributes are defined
    assert hasattr(apgpu, 'HAS_GPU')
    assert hasattr(apgpu, 'to_gpu')
    assert hasattr(apgpu, 'to_cpu')
    assert hasattr(apgpu, 'get_array_module')
    assert hasattr(apgpu, 'get_ndimage')

def test_to_gpu_to_cpu_numpy_arrays():
    arr = np.array([1, 2, 3])
    
    # Under CPU mode or GPU mode, to_cpu of a numpy array should return numpy array
    arr_cpu = apgpu.to_cpu(arr)
    assert isinstance(arr_cpu, np.ndarray)
    assert np.array_equal(arr_cpu, arr)
    
    # Test to_gpu on cpu fallback
    if not apgpu.HAS_GPU:
        arr_gpu = apgpu.to_gpu(arr)
        assert isinstance(arr_gpu, np.ndarray)
        assert np.array_equal(arr_gpu, arr)

def test_get_array_module():
    arr = np.array([1.0, 2.0, 3.0])
    xp = apgpu.get_array_module(arr)
    assert xp == np

def test_get_ndimage():
    ndimage_mod = apgpu.get_ndimage()
    if not apgpu.HAS_GPU:
        import scipy.ndimage as scipy_ndimage
        assert ndimage_mod == scipy_ndimage
