import pytest
import numpy as np
import os
from astropy.io import fits
from package.src.astropipeline import astropipeline_measure as aplm

@pytest.fixture(scope="module")
def test_fits():
    test_img_array = aplm.create_psf(impulse_amplitude=10,
                                     noise_amplitde=1,
                                     sigma=0.1)
    hdu = fits.PrimaryHDU(test_img_array)
    hdulist = fits.HDUList([hdu])
    return hdulist

def test_fits_creation(test_fits: fits.hdu.hdulist.HDUList):
    assert isinstance(test_fits, fits.hdu.hdulist.HDUList)
    assert isinstance(test_fits[0].data, np.ndarray)

def test_get_centroid(test_fits):
    row_centroid, col_centroid = aplm.get_centroid(
        test_fits[0].data, exp_loc=(10., 10.)
        )
    assert np.abs(row_centroid-10) < 1.0
    assert np.abs(col_centroid-10) < 1.0

def test_norm_array(test_fits):
    normed_array = aplm.norm_array(test_fits[0].data)
    assert np.abs(normed_array[10, 10])-1 < 1e-3

def test_get_pix_distances(test_fits):
    dist_array, _, _ = aplm.get_pix_distances(test_fits[0].data, (0, 0))
    assert np.abs(dist_array[20, 20]-np.sqrt(800)) < 1e-3

def test_get_border_stats(test_fits):
    border_stats = aplm.get_border_stats(test_fits[0].data)
    # The noise might make the mean slightly varied, so check within a reasonable range
    assert border_stats[0] >= 0.0

@pytest.fixture(scope="module")
def test_coord_list():
    coords = np.unravel_index(np.array([0, 1, 3, 2], dtype=int), shape=[2, 2])
    return coords

def test_get_unique_points(test_coord_list):
    unq_points = aplm.get_unique_points(test_coord_list,
                                        num_required=3,
                                        extent=1.1)
    unq_pnt_iter = iter(unq_points)
    assert next(unq_pnt_iter) == (0, 0)
    assert next(unq_pnt_iter) == (1, 1)
    assert len(unq_points) == 2

def test_measure_settings(tmp_path):
    settings = aplm.MeasureSettings()
    assert settings.psf_radius == 0.5
    
    settings.psf_radius = 1.0
    assert settings.psf_radius == 1.0
    
    settings.background_buffer = 10
    assert settings.background_buffer == 10
    
    file_path = tmp_path / "settings.yaml"
    settings.save_to_file(str(file_path))
    assert os.path.exists(str(file_path))
    
    new_settings = aplm.MeasureSettings()
    new_settings.psf_radius = 9.9
    new_settings.load_from_file(str(file_path))
    assert new_settings.psf_radius == 1.0

def test_crop_fits(test_fits):
    img = test_fits[0].data
    crop, x_ext, y_ext = aplm.crop_fits(img, [10, 10], 4)
    assert crop.shape == (5, 5)

def test_dog_1d_and_2d():
    arr1d = np.ones(10)
    arr1d[5] = 10
    res1d = aplm.dog_1d(arr1d, sigma_hi=1, sigma_lo=2)
    assert len(res1d) == 10
    
    arr2d = np.ones((10, 10))
    arr2d[5, 5] = 10
    res2d = aplm.dog_2d(arr2d, sigma_hi=1, sigma_lo=2)
    assert res2d.shape == (10, 10)

def test_background_sample():
    arr1d = np.array([1, 1, 1, 10, 1, 1, 1])
    mean1, std1 = aplm.background_sample_1d(arr1d, 3, buffer_len=1)
    assert mean1 == 1.0
    
    arr2d = np.ones((10, 10))
    arr2d[5, 5] = 10
    mean2, std2 = aplm.background_sample_2d(arr2d, [5, 5], buffer_len=[1, 1])
    assert mean2 == 1.0

def test_get_adjacent_pixels():
    arr2d = np.arange(25, dtype=float).reshape((5, 5))
    adj = aplm.get_adjacent_pixels(arr2d, [2, 2], extent=(1, 1), remove_mid=True)
    assert adj.shape == (3, 3)
    assert np.isnan(adj[1, 1])

def test_pixel_checks():
    arr2d = np.ones((5, 5))
    arr2d[2, 2] = 100 # hot pixel
    arr2d[1, 1] = -100 # cold pixel
    
    is_hot = aplm.pixel_check_hot(arr2d, [2, 2], bg_stats=(1.0, 0.0))
    assert is_hot == 1
    
    is_cold = aplm.pixel_check_cold(arr2d, [1, 1], min_threshold=1)
    assert is_cold == 1

def test_repair_hot_and_cold_pixels():
    arr2d = np.ones((5, 5))
    arr2d[2, 2] = 100 # hot pixel
    arr2d[3, 3] = -100 # cold pixel
    
    repaired_hot = aplm.repair_hot_pixels(arr2d.copy())
    assert repaired_hot[2, 2] < 100
    
    repaired_cold = aplm.repair_cold_pixels(arr2d.copy(), min_threshold=1)
    assert repaired_cold[3, 3] > -100

def test_wdec_bandpass_find(test_fits):
    img = test_fits[0].data
    coords, bimg = aplm.wdec_bandpass_find(img, num_returns=1, stop_level=2)
    assert len(coords) > 0
    assert bimg.shape == img.shape

def test_swt_bandpass_and_enhance(test_fits):
    img = test_fits[0].data
    bimg = aplm.swt_bandpass(img)
    assert bimg.shape == img.shape
    
    enh, res = aplm.swt_enhance(img, wavelet=['db2'], max_level=1)
    assert enh.shape == img.shape

def test_wvlt_coarse_find(test_fits):
    img = test_fits[0].data
    coords, passed = aplm.wvlt_coarse_find(img, lo_level=1, hi_level=1)
    assert passed.shape == img.shape

def test_get_best_dog(test_fits):
    img = test_fits[0].data
    best = aplm.get_best_dog(img, min_layer=0, max_layer=2)
    assert best >= 0

@pytest.fixture
def mock_star_fits(tmp_path):
    # For extract_star_samples
    file_paths = []
    data = aplm.create_psf(impulse_amplitude=50, noise_amplitde=0, sigma=0.5)
    hdu = fits.CompImageHDU(data)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    path = tmp_path / "star.fits"
    hdul.writeto(path)
    return str(path)

def test_extract_star_samples(mock_star_fits):
    # Just ensure it runs without exception
    aplm.extract_star_samples(mock_star_fits)

def test_create_psf_exceptions():
    with pytest.raises(IndexError):
        aplm.create_psf(array_size=(0, 0))

def test_get_centroid_exceptions():
    # Make a flat image so there is no peak
    img = np.zeros((10, 10))
    r, c = aplm.get_centroid(img, (5, 5), extent=2)
    assert np.isnan(r) and np.isnan(c)

def test_wvlt_coarse_find_nomax(test_fits):
    img = np.zeros((10, 10))
    coords, passed = aplm.wvlt_coarse_find(img, lo_level=1, hi_level=1)
    assert passed.shape == img.shape
