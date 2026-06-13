import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from astropy.io import fits

from package.src.astropipeline import astropipeline_manager as aplmgr


@pytest.fixture
def mock_study_df():
    df = pd.DataFrame([
        {"id": 1, "out_path": "mock_out.fits", "pipe_path": "mock_pipe.csv"}
    ])
    return df


@pytest.fixture
def mock_pipeline_df():
    df = pd.DataFrame({
        "proc_type": ["raw", "instcal", "instcal", "instcal"],
        "obs_type": ["object", "dark", "flat", "none"],
        "prod_type": ["image", "image", "image", "dqmask"],
        "url": ["url1", "url2", "url3", "url4"]
    })
    return df


@patch("package.src.astropipeline.astropipeline_manager.aple.get_pipeline_df")
@patch("package.src.astropipeline.astropipeline_manager.aple.PipeFilePaths")
@patch("package.src.astropipeline.astropipeline_manager.aplc.load_mask")
@patch("package.src.astropipeline.astropipeline_manager.aplc.get_dark_vals")
@patch("package.src.astropipeline.astropipeline_manager.aplc.get_gain_vals")
@patch("package.src.astropipeline.astropipeline_manager.aplc.image_uniformity_correct")
@patch("package.src.astropipeline.astropipeline_manager.aplc.heal_pixels")
@patch("package.src.astropipeline.astropipeline_manager.os.path.exists")
def test_correct_subpipe(
    mock_exists, mock_heal, mock_uniformity, mock_gain, mock_dark,
    mock_load_mask, mock_pipe_paths, mock_get_pipeline, mock_study_df, mock_pipeline_df
):
    # Setup mocks
    mock_exists.return_value = False  # To bypass the "exists" continue block
    mock_get_pipeline.return_value = mock_pipeline_df
    
    mock_paths = MagicMock()
    mock_paths.local_fits_path = "mock_local.fits"
    mock_paths.pipe_file_path = "mock_pipe.csv"
    mock_paths.raw_url = "mock_raw_url"
    mock_pipe_paths.return_value = mock_paths

    mock_load_mask.return_value = (MagicMock(), MagicMock())
    mock_dark.return_value = (MagicMock(), MagicMock(), [0])
    mock_gain.return_value = (MagicMock(), MagicMock(), MagicMock(), [0])
    mock_uniformity.return_value = MagicMock()
    
    mock_healed_fits = MagicMock()
    mock_heal.return_value = mock_healed_fits

    # Run correct_subpipe
    with patch("pandas.DataFrame.to_csv"):
        result_df = aplmgr.correct_subpipe(mock_study_df.copy())

    # Assertions
    assert len(result_df) == 1
    assert result_df.iloc[0]["out_path"] == "mock_local.fits"
    assert result_df.iloc[0]["pipe_path"] == "mock_pipe.csv"
    mock_heal.assert_called_once()
    mock_healed_fits.verify.assert_called_with('fix')
    mock_healed_fits.writeto.assert_called_with("mock_local.fits", overwrite=True)


@patch("package.src.astropipeline.astropipeline_manager.fits.open")
@patch("package.src.astropipeline.astropipeline_manager.WCS")
@patch("package.src.astropipeline.astropipeline_manager.aple.get_catalog_stars")
@patch("package.src.astropipeline.astropipeline_manager.SkyCoord")
@patch("package.src.astropipeline.astropipeline_manager.Cutout2D")
@patch("package.src.astropipeline.astropipeline_manager.plt")
@patch("package.src.astropipeline.astropipeline_manager.aplm.wdec_bandpass_find")
def test_undistort_subpipe(
    mock_wdec, mock_plt, mock_cutout, mock_skycoord, mock_catalog, mock_wcs, mock_fits_open, mock_study_df
):
    # Setup mocks
    # We need a primary HDU (skipped) and a CompImageHDU (processed)
    mock_primary_hdu = MagicMock(spec=fits.PrimaryHDU)
    
    mock_image_hdu = MagicMock()
    mock_image_hdu.header = {"RADESYS": "ICRS"}
    mock_image_hdu.data = np.ones((100, 100))
    
    mock_fits_open.return_value = [mock_primary_hdu, mock_image_hdu]
    
    # Mock catalog
    mock_catalog.return_value = pd.DataFrame([
        {"ra": 10.0, "dec": 10.0},
        {"ra": 20.0, "dec": 20.0}
    ])
    
    # Mock SkyCoord containment
    mock_coord_instance = MagicMock()
    mock_coord_instance.contained_by.side_effect = [False, True]
    mock_skycoord.return_value = mock_coord_instance
    
    # Mock Cutout
    mock_cutout_instance = MagicMock()
    mock_cutout_instance.data = np.ones((10, 10))
    mock_cutout.return_value = mock_cutout_instance
    
    # Mock wdec
    mock_wdec.return_value = (
        [[1, 1.0, [5, 5]]],  # found_coords (list of lists)
        np.ones((10, 10))    # img_passed
    )

    result = aplmgr.undistort_subpipe(mock_study_df)

    assert result == 0
    mock_fits_open.assert_called_once()
    mock_catalog.assert_called_once()
    assert mock_skycoord.call_count == 2
    mock_cutout.assert_called_once()
    mock_wdec.assert_called_once()
    mock_plt.show.assert_called_once()


@patch("package.src.astropipeline.astropipeline_manager.aple.get_pipeline_df")
@patch("package.src.astropipeline.astropipeline_manager.aple.PipeFilePaths")
@patch("package.src.astropipeline.astropipeline_manager.os.path.exists")
def test_correct_subpipe_cache_hit(
    mock_exists, mock_pipe_paths, mock_get_pipeline, mock_study_df, mock_pipeline_df
):
    mock_exists.return_value = True
    mock_get_pipeline.return_value = mock_pipeline_df
    mock_paths = MagicMock()
    mock_paths.local_fits_path = "mock_local.fits"
    mock_pipe_paths.return_value = mock_paths
    
    # Initialize a df where out_path is None to assert it remains unchanged (None)
    study_df = pd.DataFrame([{"id": 1, "out_path": None, "pipe_path": None}])
    result_df = aplmgr.correct_subpipe(study_df)
    assert len(result_df) == 1
    assert pd.isna(result_df.iloc[0]["out_path"])


def test_manager_main_block():
    import runpy
    
    mock_study_df = pd.DataFrame([{"id": 1, "out_path": "mock_out.fits", "pipe_path": "mock_pipe.csv"}])
    mock_pipeline_df = pd.DataFrame([{"proc_type": "raw", "obs_type": "object", "prod_type": "image", "url": "url1"}])
    
    mock_paths = MagicMock()
    mock_paths.local_fits_path = "mock_local.fits"
    
    # We will control the return value of os.path.exists using a list so we can mutate it
    exists_val = [True]
    def exists_side_effect(path):
        if "apl_study" in str(path) or "dummy" in str(path) or "dover" in str(path):
            return exists_val[0]
        return True # For fits file cache hit
        
    with patch("package.src.astropipeline.astropipeline_etl.get_study_file") as mock_get_study, \
         patch("package.src.astropipeline.astropipeline_etl.get_pipeline_df") as mock_get_pipeline, \
         patch("package.src.astropipeline.astropipeline_etl.PipeFilePaths") as mock_pipe_paths, \
         patch("package.src.astropipeline.astropipeline_etl.PipeStudy") as mock_pipestudy, \
         patch("package.src.astropipeline.astropipeline_etl.get_catalog_stars") as mock_get_stars, \
         patch("astropy.io.fits.open") as mock_fits_open, \
         patch("astropy.wcs.WCS") as mock_wcs, \
         patch("os.path.exists", side_effect=exists_side_effect) as mock_exists, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:
         
        # Mock implementations
        mock_get_study.return_value = mock_study_df.copy()
        mock_get_pipeline.return_value = mock_pipeline_df
        mock_pipe_paths.return_value = mock_paths
        mock_get_stars.return_value = pd.DataFrame() # empty catalog
        
        mock_img_hdu = MagicMock()
        mock_img_hdu.header = {"RADESYS": "ICRS"}
        mock_img_hdu.data = np.ones((10, 10))
        mock_fits_open.return_value = [fits.PrimaryHDU(), mock_img_hdu]
        
        # Case 1: study file exists
        exists_val[0] = True
        runpy.run_path(os.path.abspath("package/src/astropipeline/astropipeline_manager.py"), run_name="__main__")
        
        # Case 2: study file does not exist
        exists_val[0] = False
        mock_pipestudy_inst = MagicMock()
        mock_pipestudy_inst.find_instcals.return_value = mock_study_df.copy()
        mock_pipestudy.return_value = mock_pipestudy_inst
        
        runpy.run_path(os.path.abspath("package/src/astropipeline/astropipeline_manager.py"), run_name="__main__")

        mock_pipestudy_inst.find_instcals.assert_called_once()

