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


def test_rectify_wcs():
    data = np.ones((100, 100))
    hdu = pyfits.ImageHDU(data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 10.0
    hdu.header['CRVAL2'] = 20.0
    hdu.header['CRPIX1'] = 50.0
    hdu.header['CRPIX2'] = 50.0
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['RADESYS'] = 'ICRS'
    
    rectified_hdu = aplc.rectify_wcs(hdu)
    assert rectified_hdu is not None
    assert rectified_hdu.data.shape == (100, 100)
    assert rectified_hdu.header["CTYPE1"] == "RA---TAN"


@patch("package.src.astropipeline.astropipeline_measure.extract_star_samples")
def test_rectify_catalog(mock_extract):
    mock_extract.return_value = {
        0: [
            {'row': 30.0, 'col': 30.0, 'counts': 100.0},
            {'row': 30.0, 'col': 120.0, 'counts': 100.0},
            {'row': 120.0, 'col': 30.0, 'counts': 100.0},
            {'row': 120.0, 'col': 120.0, 'counts': 100.0},
            {'row': 75.0, 'col': 75.0, 'counts': 100.0}
        ]
    }
    
    data = np.ones((150, 150))
    hdu = pyfits.ImageHDU(data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 10.0
    hdu.header['CRVAL2'] = 20.0
    hdu.header['CRPIX1'] = 75.0
    hdu.header['CRPIX2'] = 75.0
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['RADESYS'] = 'ICRS'
    
    w = WCS(hdu.header)
    stars = []
    pixel_coords = [(30, 30), (30, 120), (120, 30), (120, 120), (75, 75)]
    for x, y in pixel_coords:
        sky = w.pixel_to_world(x, y)
        stars.append({"ra": sky.ra.deg, "dec": sky.dec.deg})
    catalog_df = pd.DataFrame(stars)
    
    log_messages = []
    def mock_log(msg):
        log_messages.append(msg)
        
    rectified_hdu = aplc.rectify_catalog(hdu, catalog_df, log_func=mock_log)
    
    assert rectified_hdu is not None
    assert rectified_hdu.data.shape == (150, 150)
    assert rectified_hdu.header["MATCHED"] == 5
    assert "MEANERR" in rectified_hdu.header


@patch("package.src.astropipeline.astropipeline_correct.rectify_catalog")
@patch("package.src.astropipeline.astropipeline_correct.WCS")
def test_rectify_image_wcs_corruption_fallback(mock_wcs, mock_rect_cat):
    mock_wcs.side_effect = Exception("Corrupted WCS header")
    
    hdu = pyfits.ImageHDU(np.ones((10, 10)))
    hdu.header['RADESYS'] = 'ICRS'
    
    catalog_df = pd.DataFrame([{"ra": 10.0, "dec": 10.0}])
    log_messages = []
    def mock_log(msg):
        log_messages.append(msg)
        
    aplc.rectify_image(hdu, method="wcs", catalog_stars_df=catalog_df, log_func=mock_log)
    
    assert any("WARNING: WCS header is corrupted" in msg for msg in log_messages)
    mock_rect_cat.assert_called_once()


@patch("package.src.astropipeline.astropipeline_measure.wdec_bandpass_find")
def test_calibrate_flux_success(mock_wdec):
    mock_wdec.return_value = (
        [[1, 1.0, ([25], [25])]],
        np.zeros((50, 50))
    )
    
    data = np.ones((50, 50)) * 100.0
    data[23:27, 23:27] = 500.0
    
    hdu = pyfits.ImageHDU(data=data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 10.0
    hdu.header['CRVAL2'] = 20.0
    hdu.header['CRPIX1'] = 25.0
    hdu.header['CRPIX2'] = 25.0
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['RADESYS'] = 'ICRS'
    
    w = WCS(hdu.header)
    stars = []
    # 9 normal stars and 1 outlier star
    for i in range(9):
        sky = w.pixel_to_world(10 + i, 10 + i)
        stars.append({"ra": sky.ra.deg, "dec": sky.dec.deg, "mag": 15.0})
    # Outlier star
    sky_outlier = w.pixel_to_world(40, 40)
    stars.append({"ra": sky_outlier.ra.deg, "dec": sky_outlier.dec.deg, "mag": 30.0})
    
    catalog_df = pd.DataFrame(stars)
    
    log_messages = []
    def mock_log(msg):
        log_messages.append(msg)
        
    cal_hdu, success = aplc.calibrate_flux(hdu, catalog_df, log_func=mock_log)
    assert success is True
    assert cal_hdu.header["BUNIT"] == "Jy"
    assert "PHOTZP" in cal_hdu.header
    assert "CALUNC" in cal_hdu.header
    assert cal_hdu.header["PHOTSYS"] == "AB"
    assert cal_hdu.header["PHOTPLAM"] == 2.2
    
    # Check that standard deviation is 0.0 (since outlier was clipped, leaving only identical 15.0 magnitudes)
    # The header keyword for zero point variance is "PHOTZPV"
    assert cal_hdu.header["PHOTZPV"] == 0.0


def test_calibrate_flux_insufficient_stars():
    data = np.ones((50, 50))
    hdu = pyfits.ImageHDU(data=data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 10.0
    hdu.header['CRVAL2'] = 20.0
    hdu.header['CRPIX1'] = 25.0
    hdu.header['CRPIX2'] = 25.0
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['RADESYS'] = 'ICRS'
    
    catalog_df = pd.DataFrame()
    log_messages = []
    def mock_log(msg):
        log_messages.append(msg)
        
    cal_hdu, success = aplc.calibrate_flux(hdu, catalog_df, log_func=mock_log)
    assert success is False
    assert cal_hdu is hdu
    assert any("WARNING: Insufficient catalog stars" in msg for msg in log_messages)


@patch("package.src.astropipeline.astropipeline_measure.extract_star_samples")
def test_calculate_global_pointing_offset(mock_extract):
    # Setup mocks for extract_star_samples for three HDUs
    # Extension 0 (no data), Extension 1, Extension 2, Extension 3
    mock_extract.side_effect = [
        {0: [{'row': 30.0, 'col': 30.0, 'counts': 100.0}]},  # HDU 1 (Extension 1 in list)
        {0: [{'row': 40.0, 'col': 40.0, 'counts': 100.0}]},  # HDU 2 (Extension 2 in list)
        {0: [{'row': 50.0, 'col': 50.0, 'counts': 100.0}]}   # HDU 3 (Extension 3 in list)
    ]
    
    hdu0 = pyfits.PrimaryHDU()  # No data
    
    hdu1 = pyfits.ImageHDU(np.ones((100, 100)))
    hdu1.header['CTYPE1'] = 'RA---TAN'
    hdu1.header['CTYPE2'] = 'DEC--TAN'
    hdu1.header['CRVAL1'] = 10.0
    hdu1.header['CRVAL2'] = 20.0
    hdu1.header['CRPIX1'] = 50.0
    hdu1.header['CRPIX2'] = 50.0
    hdu1.header['CDELT1'] = -0.0001
    hdu1.header['CDELT2'] = 0.0001
    hdu1.header['RADESYS'] = 'ICRS'
    
    hdu2 = pyfits.ImageHDU(np.ones((100, 100)))
    hdu2.header['CTYPE1'] = 'RA---TAN'
    hdu2.header['CTYPE2'] = 'DEC--TAN'
    hdu2.header['CRVAL1'] = 10.0
    hdu2.header['CRVAL2'] = 20.0
    hdu2.header['CRPIX1'] = 50.0
    hdu2.header['CRPIX2'] = 50.0
    hdu2.header['CDELT1'] = -0.0001
    hdu2.header['CDELT2'] = 0.0001
    hdu2.header['RADESYS'] = 'ICRS'
    
    hdu3 = pyfits.ImageHDU(np.ones((100, 100)))
    hdu3.header['CTYPE1'] = 'RA---TAN'
    hdu3.header['CTYPE2'] = 'DEC--TAN'
    hdu3.header['CRVAL1'] = 10.0
    hdu3.header['CRVAL2'] = 20.0
    hdu3.header['CRPIX1'] = 50.0
    hdu3.header['CRPIX2'] = 50.0
    hdu3.header['CDELT1'] = -0.0001
    hdu3.header['CDELT2'] = 0.0001
    hdu3.header['RADESYS'] = 'ICRS'
    
    hdulist = pyfits.HDUList([hdu0, hdu1, hdu2, hdu3])
    
    # We want a target offset of dx=10.0, dy=10.0 pixels
    # For hdu1: catalog star should project to (20, 20), so offset dx=30-20=10, dy=30-20=10
    # For hdu2: catalog star should project to (30, 30), so offset dx=40-30=10, dy=40-30=10
    # For hdu3: catalog star should project to (40, 40), so offset dx=50-40=10, dy=50-40=10
    w1 = WCS(hdu1.header)
    w2 = WCS(hdu2.header)
    w3 = WCS(hdu3.header)
    
    sky1 = w1.pixel_to_world(20, 20)
    sky2 = w2.pixel_to_world(30, 30)
    sky3 = w3.pixel_to_world(40, 40)
    
    catalog_df = pd.DataFrame([
        {"ra": sky1.ra.deg, "dec": sky1.dec.deg},
        {"ra": sky2.ra.deg, "dec": sky2.dec.deg},
        {"ra": sky3.ra.deg, "dec": sky3.dec.deg}
    ])
    
    dx, dy = aplc.calculate_global_pointing_offset(hdulist, catalog_df, log_func=lambda x: None)
    assert dx == 10.0
    assert dy == 10.0


@patch("package.src.astropipeline.astropipeline_measure.extract_star_samples")
def test_rectify_catalog_with_provided_offset(mock_extract):
    # Mock return value to have 5 stars, but they are offset by (10, 10)
    mock_extract.return_value = {
        0: [
            {'row': 40.0, 'col': 40.0, 'counts': 100.0},
            {'row': 40.0, 'col': 130.0, 'counts': 100.0},
            {'row': 130.0, 'col': 40.0, 'counts': 100.0},
            {'row': 130.0, 'col': 130.0, 'counts': 100.0},
            {'row': 85.0, 'col': 85.0, 'counts': 100.0}
        ]
    }
    
    data = np.ones((150, 150))
    hdu = pyfits.ImageHDU(data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 10.0
    hdu.header['CRVAL2'] = 20.0
    hdu.header['CRPIX1'] = 75.0
    hdu.header['CRPIX2'] = 75.0
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['RADESYS'] = 'ICRS'
    
    w = WCS(hdu.header)
    stars = []
    pixel_coords = [(30, 30), (30, 120), (120, 30), (120, 120), (75, 75)]
    for x, y in pixel_coords:
        sky = w.pixel_to_world(x, y)
        stars.append({"ra": sky.ra.deg, "dec": sky.dec.deg})
    catalog_df = pd.DataFrame(stars)
    
    log_messages = []
    rectified_hdu = aplc.rectify_catalog(
        hdu, catalog_df, log_func=log_messages.append, offset=(10.0, 10.0)
    )
    
    assert rectified_hdu is not None
    assert rectified_hdu.header["MATCHED"] == 5
    assert any("Using global pointing offset prior: dx=10.0, dy=10.0" in msg for msg in log_messages)





