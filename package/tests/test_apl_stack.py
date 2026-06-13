import os
import shutil
import pytest
import numpy as np
import pandas as pd
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from unittest.mock import patch, MagicMock

from package.src.astropipeline import astropipeline_correct as aplc
from package.src.astropipeline import astropipeline_stack as apls


@pytest.fixture
def temp_fits_files(tmp_path):
    paths = []
    for i in range(3):
        data = np.ones((50, 50)) * (10.0 + i)
        if i == 0:
            data[10, 10] = 100.0
            
        hdu = pyfits.PrimaryHDU(data=data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = 10.0
        hdu.header['CRVAL2'] = 20.0
        hdu.header['CRPIX1'] = 25.0
        hdu.header['CRPIX2'] = 25.0
        hdu.header['CDELT1'] = -0.0001
        hdu.header['CDELT2'] = 0.0001
        hdu.header['RADESYS'] = 'ICRS'
        
        path = tmp_path / f"frame_{i}.fits"
        hdul = pyfits.HDUList([hdu])
        hdul.writeto(path, overwrite=True)
        paths.append(str(path))
        
    return paths, str(tmp_path)


def test_fit_poly_2d():
    ny, nx = 20, 20
    y, x = np.indices((ny, nx))
    plane = 2.0 * x + 3.0 * y + 5.0
    bg = aplc.fit_poly_2d(plane, degree=1)
    assert np.allclose(plane, bg)


def test_fit_wavelet_background():
    ny, nx = 64, 64
    y, x = np.indices((ny, nx))
    bg_true = np.sin(x / 10.0) * np.cos(y / 10.0) * 1.0
    noise = np.random.normal(0, 0.05, (ny, nx))
    data = bg_true + noise
    
    bg_est = aplc.fit_wavelet_background(data, wavelet='db2', level=3)
    rmse_before = np.sqrt(np.mean((data - bg_true)**2))
    rmse_after = np.sqrt(np.mean((bg_est - bg_true)**2))
    assert rmse_after < rmse_before


def test_subtract_background():
    data = np.ones((10, 10)) * 50.0
    res_none = aplc.subtract_background(data, method="None")
    assert np.allclose(data, res_none)
    
    res_linear = aplc.subtract_background(data, method="Linear")
    assert np.allclose(res_linear, 0.0, atol=1e-5)


def test_stack_images_median_no_clip(temp_fits_files):
    paths, out_dir = temp_fits_files
    
    img_paths, dest, filename = apls.stack_images(
        image_paths=paths,
        method="median",
        sigma_clip=False,
        output_path=out_dir,
        output_filename="test_median.fits"
    )
    
    assert filename == "test_median_counts.fits"
    assert os.path.exists(dest)
    
    hdul = pyfits.open(dest)
    assert hdul[0].data[0, 0] == 11.0
    
    png_path = dest.replace(".fits", ".png")
    assert os.path.exists(png_path)
    
    log_path = os.path.join(out_dir, "stacking_pipeline.log")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        log_content = f.read()
        assert "WARNING: Stacked image was not calibrated to flux units" in log_content
        assert "Input Frames:" in log_content
    hdul.close()


def test_stack_images_mean_with_clip(temp_fits_files):
    paths, out_dir = temp_fits_files
    
    img_paths, dest, filename = apls.stack_images(
        image_paths=paths,
        method="mean",
        sigma_clip=True,
        output_path=out_dir,
        output_filename="test_mean_clip.fits"
    )
    
    assert filename == "test_mean_clip_counts.fits"
    assert os.path.exists(dest)
    
    hdul = pyfits.open(dest)
    assert hdul[0].data[0, 0] == 11.0
    assert np.isclose(hdul[0].data[10, 10], 11.5)
    hdul.close()
