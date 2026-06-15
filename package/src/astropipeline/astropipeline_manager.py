import os
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import numpy as np
import matplotlib.pyplot as plt

from package.src.astropipeline import astropipeline_correct as aplc
from package.src.astropipeline import astropipeline_etl as aple
from package.src.astropipeline import astropipeline_measure as aplm


output_folder = './fits/'
num_darks = 5
num_flats = 5
study_name = 'dover'
mask_keep_list = (0)
max_fits_results = 1

study_output_path = output_folder+'apl_study_'+study_name+'.csv'





def _correct_single_image(index, item, output_folder, study_name, num_darks, num_flats, mask_keep_list):
    pipeline_df = aple.get_pipeline_df(item)
    pipe_indexes = np.arange(0, len(pipeline_df))

    raw_index = pipe_indexes[(pipeline_df['proc_type'] == 'raw') &
                             (pipeline_df['obs_type'] == 'object')]
    raw_row = pipeline_df.iloc[raw_index]
    pipe_paths = aple.PipeFilePaths(raw_row, output_folder, study_name)

    if os.path.exists(pipe_paths.local_fits_path):
        return index, pipe_paths.local_fits_path, pipe_paths.pipe_file_path

    dqm_index = pipe_indexes[pipeline_df['prod_type'] == 'dqmask']
    dqm_row = pipeline_df.iloc[dqm_index]
    dqm_url = dqm_row['url'].iloc[0]

    pixel_mask, crop_ranges = aplc.load_mask(dqm_url, mask_keep_list)

    dark_indexes_all = pipe_indexes[pipeline_df['obs_type'] == 'dark']
    dark_urls = pipeline_df.iloc[dark_indexes_all]['url']
    dark_means, _, dark_indexes_used = aplc.get_dark_vals(dark_urls,
                                                          num_darks,
                                                          crop_ranges)
    print('Dark Cal Complete')

    flat_indexes_all = pipe_indexes[pipeline_df['obs_type'] == 'flat']
    flat_urls = pipeline_df.iloc[flat_indexes_all]['url']
    gain_vals, flat_counter, flat_times, flat_indexes_used = \
        aplc.get_gain_vals(dark_means,
                           flat_urls,
                           num_flats,
                           crop_ranges,
                           pixel_mask)

    print('Flat Cal Complete')

    print('Image Correction Starting')
    balanced_fits = aplc.image_uniformity_correct(pipe_paths.raw_url,
                                                  dark_means,
                                                  gain_vals,
                                                  crop_ranges)

    print('--Normalization Finished, Starting Mask Repair.')
    healed_fits = aplc.heal_pixels(balanced_fits,
                                   method="linear",
                                   element_select=[-1])

    print('Image Correction Finished. Saving Files.')
    healed_fits.verify('fix')
    healed_fits.writeto(pipe_paths.local_fits_path, overwrite=True)

    pipeline_df.iloc[
        np.concatenate(
            [
                raw_index,
                dark_indexes_all[dark_indexes_used],
                flat_indexes_all[flat_indexes_used],
                dqm_index
            ],
            axis=0)
        ].to_csv(pipe_paths.pipe_file_path)

    print('Output FITS saved to: '+pipe_paths.local_fits_path)
    return index, pipe_paths.local_fits_path, pipe_paths.pipe_file_path


def correct_subpipe(study_df):
    if len(study_df) <= 1:
        for index, item in study_df.iterrows():
            idx, out_path, pipe_path = _correct_single_image(
                index, item, output_folder, study_name, num_darks, num_flats, mask_keep_list
            )
            study_df.loc[idx, 'out_path'] = out_path
            study_df.loc[idx, 'pipe_path'] = pipe_path
            study_df.to_csv(study_output_path)
            print('Study details saved to: '+study_output_path)
    else:
        from concurrent.futures import ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor() as executor:
            for index, item in study_df.iterrows():
                tasks.append(
                    executor.submit(
                        _correct_single_image,
                        index,
                        item,
                        output_folder,
                        study_name,
                        num_darks,
                        num_flats,
                        mask_keep_list
                    )
                )
            for future in tasks:
                idx, out_path, pipe_path = future.result()
                study_df.loc[idx, 'out_path'] = out_path
                study_df.loc[idx, 'pipe_path'] = pipe_path
        study_df.to_csv(study_output_path)
        print('Study details saved to: '+study_output_path)
    return study_df


def _undistort_single_image(idx, study_row, method, catalog, output_folder):
    logs = []
    def log_info(message):
        logs.append(message)

    in_path = study_row['out_path']
    if not in_path or not os.path.exists(in_path):
        log_info(f"Input file {in_path} does not exist. Skipping.")
        return idx, None, logs

    log_info(f"Starting rectification for {in_path}")
    fits_in = fits.open(in_path)
    rectified_hdus = fits.HDUList()

    try:
        frame_val = 'icrs'
        for hdu in fits_in:
            if isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                frame_val = hdu.header.get('RADESYS', 'ICRS').lower()
                break
        stars_df = aple.get_catalog_stars(
            study_row,
            frame=frame_val,
            catalog=catalog
        )
    except Exception as e:
        log_info(f"Failed to query catalog stars: {str(e)}")
        stars_df = pd.DataFrame()

    global_offset = None
    if method == "catalog" and not stars_df.empty:
        try:
            global_offset = aplc.calculate_global_pointing_offset(
                fits_in,
                stars_df,
                log_func=log_info
            )
        except Exception as e:
            log_info(f"Failed to calculate global pointing offset: {str(e)}")

    for index, hdu in enumerate(fits_in):
        if isinstance(hdu, fits.hdu.image.PrimaryHDU) and (hdu.data is None or hdu.data.size == 0):
            try:
                rectified_hdus.append(hdu.copy())
            except Exception:
                try:
                    rectified_hdus.append(hdu)
                except Exception:
                    pass
            continue

        if isinstance(hdu.data, np.ndarray):
            try:
                rect_hdu = aplc.rectify_image(
                    hdu,
                    method=method,
                    catalog_stars_df=stars_df,
                    log_func=log_info,
                    offset=global_offset
                )
                rectified_hdus.append(rect_hdu)
            except Exception as e:
                log_info(f"Rectification failed for extension {index}: {str(e)}")
                try:
                    rectified_hdus.append(hdu.copy())
                except Exception:
                    try:
                        rectified_hdus.append(hdu)
                    except Exception:
                        pass
        else:
            try:
                rectified_hdus.append(hdu.copy())
            except Exception:
                try:
                    rectified_hdus.append(hdu)
                except Exception:
                    pass

    if "_rectified" in in_path:
        rectified_path = in_path
    else:
        base, ext = os.path.splitext(in_path)
        if ext == ".fz":
            base2, ext2 = os.path.splitext(base)
            rectified_path = base2 + "_rectified.fits"
        else:
            rectified_path = base + "_rectified.fits"

    rectified_hdus.writeto(rectified_path, overwrite=True, output_verify="ignore")
    if hasattr(fits_in, "close"):
        fits_in.close()
    rectified_hdus.close()

    log_info(f"Saved rectified image to: {rectified_path}")
    return idx, rectified_path, logs


def undistort_subpipe(study_df, method="wcs", catalog="2MASS"):
    def log_info(message):
        print(message)
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "pipeline.log"), "a") as f:
            f.write(message + "\n")

    if len(study_df) <= 1:
        for idx, study_row in study_df.iterrows():
            _, rectified_path, logs = _undistort_single_image(
                idx, study_row, method, catalog, output_folder
            )
            for msg in logs:
                log_info(msg)
            if rectified_path:
                study_df.loc[idx, 'rectified_path'] = rectified_path
    else:
        from concurrent.futures import ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor() as executor:
            for idx, study_row in study_df.iterrows():
                tasks.append(
                    executor.submit(
                        _undistort_single_image,
                        idx,
                        study_row,
                        method,
                        catalog,
                        output_folder
                    )
                )
            for future in tasks:
                idx, rectified_path, logs = future.result()
                for msg in logs:
                    log_info(msg)
                if rectified_path:
                    study_df.loc[idx, 'rectified_path'] = rectified_path
    return study_df


def _calibrate_single_image(idx, study_row, catalog, output_folder):
    logs = []
    def log_info(message):
        logs.append(message)

    in_path = study_row.get('rectified_path') or study_row.get('out_path')
    if not in_path or not os.path.exists(in_path):
        log_info(f"Input file {in_path} does not exist. Skipping.")
        return idx, None, logs

    log_info(f"Starting flux calibration for {in_path}")
    fits_in = fits.open(in_path)
    calibrated_hdus = fits.HDUList()

    for index, hdu in enumerate(fits_in):
        if isinstance(hdu, (fits.hdu.image.PrimaryHDU, fits.PrimaryHDU)) and (hdu.data is None or hdu.data.size == 0):
            try:
                calibrated_hdus.append(hdu.copy())
            except Exception:
                try:
                    calibrated_hdus.append(hdu)
                except Exception:
                    pass
            continue

        if isinstance(hdu.data, np.ndarray):
            try:
                stars_df = aple.get_catalog_stars(
                    study_row,
                    frame=hdu.header.get('RADESYS', 'ICRS').lower(),
                    catalog=catalog)
            except Exception as e:
                log_info(f"Failed to query catalog stars: {str(e)}")
                stars_df = pd.DataFrame()

            try:
                cal_hdu, success = aplc.calibrate_flux(
                    hdu,
                    catalog_stars_df=stars_df,
                    log_func=log_info
                )
                calibrated_hdus.append(cal_hdu)
            except Exception as e:
                log_info(f"Flux calibration failed for extension {index}: {str(e)}")
                try:
                    calibrated_hdus.append(hdu.copy())
                except Exception:
                    try:
                        calibrated_hdus.append(hdu)
                    except Exception:
                        pass
        else:
            try:
                calibrated_hdus.append(hdu.copy())
            except Exception:
                try:
                    calibrated_hdus.append(hdu)
                except Exception:
                    pass

    if "_calibrated" in in_path:
        calibrated_path = in_path
    else:
        base, ext = os.path.splitext(in_path)
        if ext == ".fz":
            base2, ext2 = os.path.splitext(base)
            calibrated_path = base2 + "_calibrated.fits"
        else:
            calibrated_path = base + "_calibrated.fits"

    calibrated_hdus.writeto(calibrated_path, overwrite=True, output_verify="ignore")
    if hasattr(fits_in, "close"):
        fits_in.close()
    calibrated_hdus.close()

    log_info(f"Saved calibrated image to: {calibrated_path}")
    return idx, calibrated_path, logs


def calibrate_flux_subpipe(study_df, catalog="2MASS"):
    def log_info(message):
        print(message)
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "pipeline.log"), "a") as f:
            f.write(message + "\n")

    if len(study_df) <= 1:
        for idx, study_row in study_df.iterrows():
            _, calibrated_path, logs = _calibrate_single_image(
                idx, study_row, catalog, output_folder
            )
            for msg in logs:
                log_info(msg)
            if calibrated_path:
                study_df.loc[idx, 'calibrated_path'] = calibrated_path
    else:
        from concurrent.futures import ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor() as executor:
            for idx, study_row in study_df.iterrows():
                tasks.append(
                    executor.submit(
                        _calibrate_single_image,
                        idx,
                        study_row,
                        catalog,
                        output_folder
                    )
                )
            for future in tasks:
                idx, calibrated_path, logs = future.result()
                for msg in logs:
                    log_info(msg)
                if calibrated_path:
                    study_df.loc[idx, 'calibrated_path'] = calibrated_path
    return study_df


def study_single_object(study_df, ra, dec, crop_size=75, catalog="2MASS", method="catalog", output_folder=output_folder, study_name=study_name):
    """
    Perform standard correction, rectification, and calibration on each frame,
    then crop a given pixel-based area (default 75x75 pixels) around the object (specified by ra, dec).
    
    Parameters:
    -----------
    study_df : pandas.DataFrame
        DataFrame representing the observations (from PipeStudy.find_instcals()).
    ra : float
        Right Ascension of the target object in degrees.
    dec : float
        Declination of the target object in degrees.
    crop_size : int or tuple/list of int, default 75
        Size of the crop in pixels. If an integer is provided, crops a square of shape (crop_size, crop_size).
    catalog : str, default "2MASS"
        Catalog to use for rectification/flux calibration.
    method : str, default "catalog"
        Method for rectification ('catalog' or 'wcs').
    output_folder : str, default output_folder
        Output directory folder.
    study_name : str, default study_name
        Study name to append to study files.
        
    Returns:
    --------
    cropped_paths : list of str
        List of paths to the saved cropped FITS files.
    """
    # 1. Run correction
    study_df = correct_subpipe(study_df)
    
    # 2. Run rectification
    study_df = undistort_subpipe(study_df, method=method, catalog=catalog)
    
    # 3. Run calibration
    study_df = calibrate_flux_subpipe(study_df, catalog=catalog)
    
    # Ensure crop_size is a tuple of (ny, nx)
    if isinstance(crop_size, int):
        size = (crop_size, crop_size)
    else:
        size = tuple(crop_size)
        
    coord = SkyCoord(ra, dec, unit="deg")
    cropped_paths = []
    
    for idx, row in study_df.iterrows():
        in_path = row.get("calibrated_path") or row.get("rectified_path") or row.get("out_path")
        if not in_path or not os.path.exists(in_path):
            print(f"Preprocessed file {in_path} does not exist. Skipping crop.")
            continue
            
        base, ext = os.path.splitext(in_path)
        crop_path = f"{base}_crop{ext}"
        
        fits_in = fits.open(in_path)
        hdus_cropped = fits.HDUList()
        
        has_crop = False
        
        for index, hdu in enumerate(fits_in):
            if isinstance(hdu, (fits.hdu.image.PrimaryHDU, fits.PrimaryHDU)) and (hdu.data is None or hdu.data.size == 0):
                hdus_cropped.append(hdu.copy())
                continue
                
            if isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                wcs = WCS(hdu.header)
                try:
                    # Perform crop using Cutout2D
                    cutout = Cutout2D(hdu.data, coord, size, wcs=wcs, mode='partial', fill_value=np.nan)
                    
                    # Create new HDU for cropped data
                    if isinstance(hdu, fits.PrimaryHDU):
                        new_hdu = fits.PrimaryHDU(data=cutout.data, header=hdu.header.copy())
                    else:
                        new_hdu = fits.ImageHDU(data=cutout.data, header=hdu.header.copy())
                        
                    # Update header with cutout WCS
                    new_hdu.header.update(cutout.wcs.to_header())
                    hdus_cropped.append(new_hdu)
                    has_crop = True
                except Exception as e:
                    # Discard if it does not overlap or fails to crop
                    pass
            else:
                pass
                
        fits_in.close()
        
        if has_crop:
            hdus_cropped.writeto(crop_path, overwrite=True, output_verify="ignore")
            hdus_cropped.close()
            study_df.loc[idx, 'cropped_path'] = crop_path
            cropped_paths.append(crop_path)
        else:
            hdus_cropped.close()
            print(f"Warning: frame {in_path} does not contain target coordinate. Discarding from study.")
            
    # Write updated study_df
    study_df.to_csv(study_output_path)
    print(f"Study details with crops saved to: {study_output_path}")
    return cropped_paths


if __name__ == '__main__':
    if os.path.exists(study_output_path):
        test_study_df = aple.get_study_file(study_output_path)
    else:
        test_pipe_study = aple.PipeStudy(telescope="kp4m",
                                         instrument="newfirm",
                                         exposure=10,
                                         filter="KXs",
                                         max_returns=max_fits_results)
        test_study_df = test_pipe_study.find_instcals()
        test_study_df.to_csv(study_output_path)

    test_study_df = correct_subpipe(test_study_df)

    test_study_df = aple.get_study_file(study_output_path)

    test_study_df = undistort_subpipe(test_study_df)

    test_study_df = calibrate_flux_subpipe(test_study_df)
