import os
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from package.src.astropipeline import astropipeline_correct as aplc


@pytest.fixture
def sample_fits_image():
    # Create a simple fits image with a NaN pixel to test heal_pixels
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, np.nan, 6.0],
                     [7.0, 8.0, 9.0]])
    hdu = pyfits.PrimaryHDU(data)
    return pyfits.HDUList([hdu])


def test_heal_pixels_mean(sample_fits_image):
    healed = aplc.heal_pixels(sample_fits_image, method="mean", element_select=[0])
    data = healed[0].data
    assert not np.isnan(data[1, 1])
    # The mean of [1,2,3,4,6,7,8,9] is 40 / 8 = 5.0
    assert data[1, 1] == 5.0


def test_heal_pixels_linear(sample_fits_image):
    healed = aplc.heal_pixels(sample_fits_image, method="linear", element_select=[0])
    data = healed[0].data
    assert not np.isnan(data[1, 1])
    # Linear interpolation at the center should be 5.0
    assert np.isclose(data[1, 1], 5.0)


def test_heal_pixels_quadratic(sample_fits_image):
    # The quadratic method just falls back to linear and prints a warning
    healed = aplc.heal_pixels(sample_fits_image, method="quadratic", element_select=[-1])
    data = healed[0].data
    assert not np.isnan(data[1, 1])
    assert np.isclose(data[1, 1], 5.0)


@pytest.fixture
def mock_dqm_fits(tmp_path):
    # Create a compressed image HDU for load_mask
    data = np.array([[0, 1], [1, 0]], dtype=np.int16)
    hdu = pyfits.CompImageHDU(data)
    hdul = pyfits.HDUList([pyfits.PrimaryHDU(), hdu])
    
    file_path = tmp_path / "mock_dqm.fits"
    hdul.writeto(file_path)
    return str(file_path)


def test_load_mask(mock_dqm_fits):
    pixel_mask, crop_ranges = aplc.load_mask(mock_dqm_fits, keep_list=(0,))
    
    assert 1 in pixel_mask
    assert 1 in crop_ranges
    
    mask = pixel_mask[1]
    # keep_list is (0,), so elements that are NOT 0 should be True (masked)
    assert mask[0, 0] == False
    assert mask[0, 1] == True
    assert mask[1, 0] == True
    assert mask[1, 1] == False


@pytest.fixture
def mock_dark_fits(tmp_path):
    # Create mock dark FITS
    file_paths = []
    for i in range(2):
        data = np.ones((2, 2)) * (i + 1) * 10 # 10 and 20
        hdu = pyfits.CompImageHDU(data)
        hdul = pyfits.HDUList([pyfits.PrimaryHDU(), hdu])
        path = tmp_path / f"dark_{i}.fits"
        hdul.writeto(path)
        file_paths.append(str(path))
    return pd.Series(file_paths)


@patch('package.src.astropipeline.astropipeline_correct.random.choices')
def test_get_dark_vals(mock_choices, mock_dark_fits):
    # Mock random.choices to just return [0, 1]
    mock_choices.return_value = [0, 1]
    
    crop_ranges = {1: np.ix_([0, 1], [0, 1])}
    dark_means, dark_counters, urls = aplc.get_dark_vals(mock_dark_fits, 2, crop_ranges)
    
    assert 1 in dark_means
    assert dark_counters[1] == 2
    # The sum should be 10 + 20 = 30, mean = 15
    assert dark_means[1][0, 0] == 15.0


@pytest.fixture
def mock_flat_fits(tmp_path):
    file_paths = []
    for i in range(2):
        data = np.ones((2, 2)) * 100
        hdu = pyfits.CompImageHDU(data)
        hdu.header['EXPTIME'] = 1.0
        hdul = pyfits.HDUList([pyfits.PrimaryHDU(), hdu])
        path = tmp_path / f"flat_{i}.fits"
        hdul.writeto(path)
        file_paths.append(str(path))
    return pd.Series(file_paths)


@patch('package.src.astropipeline.astropipeline_correct.random.choices')
def test_get_gain_vals(mock_choices, mock_flat_fits):
    mock_choices.return_value = [0, 1]
    
    dark_means = {1: np.ones((2, 2)) * 10}
    crop_ranges = {1: np.ix_([0, 1], [0, 1])}
    pixel_mask = {1: np.array([[False, False], [False, True]])}
    
    gain_vals, counters, cumtime, urls = aplc.get_gain_vals(
        dark_means, mock_flat_fits, 2, crop_ranges, pixel_mask
    )
    
    assert 1 in gain_vals
    assert counters[1] == 2
    assert cumtime[1] == 2.0
    
    # Flats are 100. Dark is 10. Value before mean = 90. Mean = 90.
    # Mask is True at [1, 1], so it should be NaN there.
    assert gain_vals[1][0, 0] == 90.0
    assert np.isnan(gain_vals[1][1, 1])


def test_parse_wat_table():
    hdu = pyfits.PrimaryHDU()
    hdu.header['NAXIS'] = 1
    hdu.header['WAT1_001'] = 'wtype=linear axtype=ra '
    hdu.header['WAT1_002'] = ' lngcor = "test_lng" '
    
    wat_df = aplc.parse_wat_table(hdu)
    assert len(wat_df) == 1
    assert wat_df.iloc[0]["wtype"] == "linear"
    assert wat_df.iloc[0]["axtype"] == "ra"
    assert wat_df.iloc[0]["dc_vals"] == "test_lng"


@patch("package.src.astropipeline.astropipeline_correct.pyfits.open")
def test_image_uniformity_correct(mock_fits_open, tmp_path):
    # Ensure fits directory exists
    os.makedirs("./fits", exist_ok=True)
    
    # Mocking fits HDU list
    hdu0 = pyfits.PrimaryHDU()
    hdu0.header["RADECSYS"] = "ICRS"
    
    # Create an image HDU with dummy WCS and data
    hdu1 = pyfits.ImageHDU(np.ones((10, 10)) * 20.0)
    # Give it some basic WCS keywords to not fail
    hdu1.header['CTYPE1'] = 'RA---TAN'
    hdu1.header['CTYPE2'] = 'DEC--TAN'
    hdu1.header['CRVAL1'] = 0.0
    hdu1.header['CRVAL2'] = 0.0
    hdu1.header['CRPIX1'] = 5.0
    hdu1.header['CRPIX2'] = 5.0
    hdu1.header['CDELT1'] = 1.0
    hdu1.header['CDELT2'] = 1.0
    hdu1.header['WAT1_001'] = 'wtype=linear axtype=ra lngcor = "test_lng"'
    hdu1.header['WAT2_001'] = 'wtype=linear axtype=dec latcor = "test_lat"'
    
    mock_hdulist = pyfits.HDUList([
        hdu0,
        pyfits.ImageHDU(),
        pyfits.ImageHDU(),
        pyfits.ImageHDU(),
        hdu1
    ])
    # For updating the temp file
    mock_hdulist.copy = MagicMock(return_value=mock_hdulist)
    
    # Context manager mock
    mock_fits_open.return_value.__enter__.return_value = mock_hdulist
    mock_fits_open.return_value = mock_hdulist

    dark_means = {4: 10.0}
    gain_vals = {4: np.ones((10, 10)) * 2.0}
    crop_ranges = {4: np.ix_(range(10), range(10))}
    
    balanced = aplc.image_uniformity_correct("mock_url", dark_means, gain_vals, crop_ranges)
    
    # Data is (20.0 - 10.0) / 2.0 = 5.0
    assert balanced[4].data[0, 0] == 5.0
    assert "RADESYSa" in balanced[0].header


@patch("package.src.astropipeline.astropipeline_correct.pyfits.open")
@patch("package.src.astropipeline.astropipeline_correct.WCS")
def test_image_uniformity_correct_wcs_error(mock_wcs, mock_fits_open, tmp_path):
    os.makedirs("./fits", exist_ok=True)
    mock_wcs.side_effect = Exception("Mock WCS error")
    
    hdu0 = pyfits.PrimaryHDU()
    hdu0.header["RADECSYS"] = "ICRS"
    
    hdu1 = pyfits.ImageHDU(np.ones((10, 10)) * 20.0)
    hdu1.header['WAT1_001'] = 'wtype=linear axtype=ra lngcor = "test_lng"'
    hdu1.header['WAT2_001'] = 'wtype=linear axtype=dec latcor = "test_lat"'
    
    mock_hdulist = pyfits.HDUList([
        hdu0,
        pyfits.ImageHDU(),
        pyfits.ImageHDU(),
        pyfits.ImageHDU(),
        hdu1
    ])
    mock_hdulist.copy = MagicMock(return_value=mock_hdulist)
    
    mock_fits_open.return_value.__enter__.return_value = mock_hdulist
    mock_fits_open.return_value = mock_hdulist

    dark_means = {4: 10.0}
    gain_vals = {4: np.ones((10, 10)) * 2.0}
    crop_ranges = {4: np.ix_(range(10), range(10))}
    
    balanced = aplc.image_uniformity_correct("mock_url", dark_means, gain_vals, crop_ranges)
    assert len(balanced) == 4


