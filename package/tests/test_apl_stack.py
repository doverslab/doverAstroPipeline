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


def test_stack_images_with_output_stats(temp_fits_files):
    paths, out_dir = temp_fits_files
    
    img_paths, dest, filename = apls.stack_images(
        image_paths=paths,
        method="median",
        sigma_clip=False,
        output_path=out_dir,
        output_filename="test_stats.fits",
        output_stats=True
    )
    
    assert filename == "test_stats_counts.fits"
    assert os.path.exists(dest)
    
    stats_png_path = dest.replace(".fits", "_stats.png")
    assert os.path.exists(stats_png_path)
    
    log_path = os.path.join(out_dir, "stacking_pipeline.log")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        log_content = f.read()
        assert "Stats Location:" in log_content
        assert stats_png_path in log_content


def test_stack_images_with_nans(tmp_path):
    # Create fits files with some NaNs
    paths = []
    for i in range(3):
        data = np.ones((10, 10)) * (10.0 + i)
        data[0, 0] = np.nan
        data[1, 1] = np.nan
        if i == 0:
            data[2, 2] = np.nan  # NaN in only one frame
            
        hdu = pyfits.PrimaryHDU(data=data)
        path = tmp_path / f"nan_frame_{i}.fits"
        hdul = pyfits.HDUList([hdu])
        hdul.writeto(path, overwrite=True)
        paths.append(str(path))

    # Test median stacking without sigma clip
    _, dest, _ = apls.stack_images(
        image_paths=paths,
        method="median",
        sigma_clip=False,
        output_path=str(tmp_path),
        output_filename="nan_median.fits"
    )
    hdul = pyfits.open(dest)
    data = hdul[0].data
    assert np.isnan(data[0, 0])  # all frames had NaN here, so stacked value is NaN
    assert np.isnan(data[1, 1])  # all frames had NaN here, so stacked value is NaN
    assert np.isclose(data[2, 2], 11.5)
    hdul.close()

    # Test mean stacking with sigma clip
    _, dest_clip, _ = apls.stack_images(
        image_paths=paths,
        method="mean",
        sigma_clip=True,
        output_path=str(tmp_path),
        output_filename="nan_mean_clip.fits"
    )
    hdul = pyfits.open(dest_clip)
    data = hdul[0].data
    assert np.isnan(data[0, 0])
    assert np.isnan(data[1, 1])
    assert np.isclose(data[2, 2], 11.5)
    hdul.close()


def test_simple_hist_nans(tmp_path):
    from package.src.astropipeline import astropipeline_measure as aplm
    
    # 1. Partial NaNs and Infs
    data = np.array([[1.0, 2.0, np.nan], [4.0, np.inf, 6.0], [-np.inf, np.nan, 8.0]])
    save_path = tmp_path / "hist_partial_nan.png"
    counts, centers = aplm.simple_hist(data, nbins=5, save_path=str(save_path))
    assert len(counts) == 5
    assert os.path.exists(save_path)
    
    # 2. All NaNs and Infs
    data_all_nan = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
    save_path_all = tmp_path / "hist_all_nan.png"
    counts_all, centers_all = aplm.simple_hist(data_all_nan, nbins=5, save_path=str(save_path_all))
    assert len(counts_all) == 5
    assert np.all(counts_all == 0)
    assert os.path.exists(save_path_all)


def test_check_gaussian_chi2():
    from package.src.astropipeline.astropipeline_stack import check_gaussian_chi2
    
    # 1. Normally distributed data -> should return True
    np.random.seed(42)
    gaussian_data = np.random.normal(loc=10.0, scale=2.0, size=(100, 100))
    assert check_gaussian_chi2(gaussian_data, alpha=0.01) is True
    
    # 2. Uniformly distributed data -> should return False
    uniform_data = np.random.uniform(low=0.0, high=10.0, size=(100, 100))
    assert check_gaussian_chi2(uniform_data, alpha=0.01) is False
    
    # 3. Insufficient data (< 20 elements) -> should return True (failsafe)
    small_data = np.random.uniform(low=0.0, high=10.0, size=(3, 3))
    assert check_gaussian_chi2(small_data, alpha=0.01) is True


def test_stack_images_reject_non_gaussian(tmp_path):
    # Create two FITS files with Gaussian distribution
    # and one FITS file with a non-Gaussian distribution
    paths = []
    np.random.seed(42)
    
    # Gaussian frame 0
    data_g0 = np.random.normal(loc=10.0, scale=1.0, size=(50, 50))
    hdu_g0 = pyfits.PrimaryHDU(data=data_g0)
    path_g0 = tmp_path / "gaussian_frame_0.fits"
    pyfits.HDUList([hdu_g0]).writeto(path_g0, overwrite=True)
    paths.append(str(path_g0))
    
    # Gaussian frame 1
    data_g1 = np.random.normal(loc=10.0, scale=1.0, size=(50, 50))
    hdu_g1 = pyfits.PrimaryHDU(data=data_g1)
    path_g1 = tmp_path / "gaussian_frame_1.fits"
    pyfits.HDUList([hdu_g1]).writeto(path_g1, overwrite=True)
    paths.append(str(path_g1))
    
    # Non-Gaussian frame (Uniform distribution)
    data_ng = np.random.uniform(low=0.0, high=20.0, size=(50, 50))
    hdu_ng = pyfits.PrimaryHDU(data=data_ng)
    path_ng = tmp_path / "nongaussian_frame.fits"
    pyfits.HDUList([hdu_ng]).writeto(path_ng, overwrite=True)
    paths.append(str(path_ng))
    
    # Stack with reject_non_gaussian=True
    accepted_paths, dest, filename = apls.stack_images(
        image_paths=paths,
        method="median",
        sigma_clip=False,
        output_path=str(tmp_path),
        output_filename="test_reject.fits",
        reject_non_gaussian=True
    )
    
    # Verify that only the 2 Gaussian frames were stacked
    assert len(accepted_paths) == 2
    assert str(path_g0) in accepted_paths
    assert str(path_g1) in accepted_paths
    assert str(path_ng) not in accepted_paths
    
    # Verify that the stacked FITS file exists
    assert os.path.exists(dest)
    
    # Verify that raising error works if all are rejected
    with pytest.raises(ValueError, match="All input frames were rejected"):
        apls.stack_images(
            image_paths=[str(path_ng)],
            method="median",
            sigma_clip=False,
            output_path=str(tmp_path),
            output_filename="test_reject_all.fits",
            reject_non_gaussian=True
        )


@patch("package.src.astropipeline.astropipeline_manager.study_single_object")
@patch("package.src.astropipeline.astropipeline_stack.stack_images")
@patch("package.src.astropipeline.astropipeline_etl.get_catalog_stars")
@patch("package.src.astropipeline.astropipeline_stack.fits.open")
def test_stack_single_object(mock_fits_open, mock_get_catalog, mock_stack_images, mock_study_single):
    study_df = pd.DataFrame([{"id": 1, "out_path": "frame1.fits"}])
    
    mock_study_single.return_value = ["frame1_crop.fits"]
    mock_stack_images.return_value = (["frame1_crop.fits"], "stacked_crop.fits", "stacked_crop_counts.fits")
    
    hdr = pyfits.Header()
    hdr["RADESYS"] = "ICRS"
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 10.0
    hdr["CRVAL2"] = 20.0
    hdr["CRPIX1"] = 5.0
    hdr["CRPIX2"] = 5.0
    hdr["CDELT1"] = -0.0001
    hdr["CDELT2"] = 0.0001
    
    mock_hdu = pyfits.PrimaryHDU(data=np.ones((10, 10)), header=hdr)
    mock_hdulist = pyfits.HDUList([mock_hdu])
    mock_fits_open.return_value = mock_hdulist
    
    mock_get_catalog.return_value = pd.DataFrame([{"ra": 10.0, "dec": 20.0, "Kmag": 12.0}])
    
    accepted, dest, filename = apls.stack_single_object(
        study_df=study_df,
        ra=10.0,
        dec=20.0,
        crop_size=10,
        output_path="fits/"
    )
    
    assert accepted == ["frame1_crop.fits"]
    assert dest == "stacked_crop.fits"
    assert filename == "stacked_crop_counts.fits"
    
    mock_study_single.assert_called_once_with(
        study_df=study_df,
        ra=10.0,
        dec=20.0,
        crop_size=10,
        catalog="2MASS",
        method="catalog",
        output_folder="fits/"
    )
    mock_stack_images.assert_called_once()




