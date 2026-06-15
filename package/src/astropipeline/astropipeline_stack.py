import os
import time
import numpy as np
import pandas as pd
import astropy.io.fits as fits
import matplotlib
try:
    from IPython import get_ipython
    if get_ipython() is None:
        matplotlib.use('Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package.src.astropipeline import astropipeline_correct as aplc
from package.src.astropipeline import astropipeline_gpu as apgpu
from package.src.astropipeline import astropipeline_measure as aplm
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D


def _process_single_frame(path, final_cal, cal_frames_flux, catalog_stars_df, bg_sub_method):
    hdul = fits.open(path)
    img_hdu = None
    for idx, hdu in enumerate(hdul):
        if isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
            img_hdu = hdu.copy()
            break
    if img_hdu is None:
        img_hdu = hdul[0].copy()
        
    hdul.close()
    
    if final_cal == "Master bias and non-uniformity correction":
        pass
        
    local_calibrated_success = False
    if cal_frames_flux:
        try:
            if catalog_stars_df is not None and not catalog_stars_df.empty:
                img_hdu, success = aplc.calibrate_flux(img_hdu, catalog_stars_df)
                if success:
                    local_calibrated_success = True
            else:
                raise ValueError("No catalog stars provided for frame calibration.")
        except Exception:
            pass
            
    if isinstance(img_hdu.data, np.ndarray):
        img_hdu.data = aplc.subtract_background(img_hdu.data, method=bg_sub_method)
        
    return img_hdu, local_calibrated_success


def check_gaussian_chi2(data, alpha=0.05):
    """
    Check if the pixel values in data follow a Gaussian distribution
    using a Chi-Squared goodness-of-fit test.
    """
    import scipy.stats as stats
    clean_data = data[np.isfinite(data)]
    if len(clean_data) < 20:
        # Insufficient data to reject normality
        return True
        
    mean = np.mean(clean_data)
    std = np.std(clean_data)
    if std == 0:
        return False
        
    n_bins = 10
    quantiles = np.linspace(0, 1, n_bins + 1)
    
    bin_edges = stats.norm.ppf(quantiles, loc=mean, scale=std)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    observed, _ = np.histogram(clean_data, bins=bin_edges)
    expected = np.full(n_bins, len(clean_data) / n_bins)
    
    chi2_stat, p_val = stats.chisquare(observed, f_exp=expected, ddof=2)
    return bool(p_val >= alpha)


def stack_images(
    image_paths,
    method="median",
    sigma_clip=False,
    bg_sub_method="None",
    final_cal="None",
    cal_stacked_flux=False,
    cal_frames_flux=False,
    catalog_stars_df=None,
    output_path="./fits/",
    output_filename="stacked.fits",
    output_stats=False,
    reject_non_gaussian=False
):
    start_time = time.time()
    steps_timing = {}
    
    # 1. Calibration / Load frames / Background Subtraction (Parallelized)
    t_start_cal = time.time()
    processed_hdus = []
    frame_calibrated_success = False
    
    if len(image_paths) <= 1:
        for path in image_paths:
            img_hdu, success = _process_single_frame(
                path, final_cal, cal_frames_flux, catalog_stars_df, bg_sub_method
            )
            processed_hdus.append(img_hdu)
            if success:
                frame_calibrated_success = True
        steps_timing["Calibration"] = time.time() - t_start_cal
        steps_timing["Background subtraction"] = 0.0
    else:
        from concurrent.futures import ProcessPoolExecutor
        tasks = []
        with ProcessPoolExecutor() as executor:
            for path in image_paths:
                tasks.append(
                    executor.submit(
                        _process_single_frame,
                        path,
                        final_cal,
                        cal_frames_flux,
                        catalog_stars_df,
                        bg_sub_method
                    )
                )
            for future in tasks:
                img_hdu, success = future.result()
                processed_hdus.append(img_hdu)
                if success:
                    frame_calibrated_success = True
        steps_timing["Calibration"] = time.time() - t_start_cal
        steps_timing["Background subtraction"] = 0.0

    # 2. Gaussian Normality Check & Rejection
    accepted_hdus = []
    accepted_paths = []
    rejected_paths = []
    
    for path, hdu in zip(image_paths, processed_hdus):
        if reject_non_gaussian:
            if isinstance(hdu.data, np.ndarray):
                is_normal = check_gaussian_chi2(hdu.data)
            else:
                is_normal = False
            
            if is_normal:
                accepted_hdus.append(hdu)
                accepted_paths.append(path)
            else:
                rejected_paths.append(path)
        else:
            accepted_hdus.append(hdu)
            accepted_paths.append(path)
            
    if reject_non_gaussian:
        print(f"Normality check: accepted {len(accepted_hdus)}/{len(processed_hdus)} frames.")
        if rejected_paths:
            print(f"Rejected non-Gaussian frames: {rejected_paths}")
            
    if not accepted_hdus:
        raise ValueError("All input frames were rejected because their pixel values do not have a Gaussian distribution.")
        
    processed_hdus = accepted_hdus

    
    # 3. Sigma Clipping
    t_start_sigma = time.time()
    data_stack = np.array([hdu.data for hdu in processed_hdus if isinstance(hdu.data, np.ndarray)])
    
    if apgpu.HAS_GPU:
        try:
            cp = apgpu.cp
            data_stack_cp = cp.asarray(data_stack)
            if sigma_clip and len(data_stack_cp) > 1:
                threshold = 3.0
                if isinstance(sigma_clip, (int, float)) and not isinstance(sigma_clip, bool):
                    threshold = sigma_clip
                elif len(data_stack_cp) < 5:
                    threshold = 1.0
                    
                mean = cp.nanmean(data_stack_cp, axis=0)
                std = cp.nanstd(data_stack_cp, axis=0)
                std[std == 0] = 1e-10
                deviations = cp.abs(data_stack_cp - mean) / std
                clip_mask = (deviations > threshold) | cp.isnan(data_stack_cp)
                data_stack_clipped = cp.where(clip_mask, cp.nan, data_stack_cp)
            else:
                data_stack_clipped = cp.where(cp.isnan(data_stack_cp), cp.nan, data_stack_cp)
            
            steps_timing["Sigma clipping"] = time.time() - t_start_sigma
            
            # 4. Stacking
            t_start_stack = time.time()
            if method == "mean":
                stacked_data_cp = cp.nanmean(data_stack_clipped, axis=0)
            else:
                stacked_data_cp = cp.nanmedian(data_stack_clipped, axis=0)
            
            stacked_data = cp.asnumpy(stacked_data_cp)
            steps_timing["Stacking"] = time.time() - t_start_stack
            use_gpu_success = True
        except Exception as e:
            print(f"GPU stacking failed, falling back to CPU: {str(e)}")
            use_gpu_success = False
    else:
        use_gpu_success = False
        
    if not use_gpu_success:
        if sigma_clip and len(data_stack) > 1:
            threshold = 3.0
            if isinstance(sigma_clip, (int, float)) and not isinstance(sigma_clip, bool):
                threshold = sigma_clip
            elif len(data_stack) < 5:
                threshold = 1.0
                
            mean = np.nanmean(data_stack, axis=0)
            std = np.nanstd(data_stack, axis=0)
            std[std == 0] = 1e-10
            deviations = np.abs(data_stack - mean) / std
            mask = (deviations > threshold) | np.isnan(data_stack)
            data_stack_masked = np.ma.masked_array(data_stack, mask=mask)
        else:
            data_stack_masked = np.ma.masked_array(data_stack, mask=np.isnan(data_stack))
            
        steps_timing["Sigma clipping"] = time.time() - t_start_sigma
        
        # 4. Stacking
        t_start_stack = time.time()
        if method == "mean":
            stacked_data = np.ma.mean(data_stack_masked, axis=0).filled(np.nan)
        else:
            stacked_data = np.ma.median(data_stack_masked, axis=0).filled(np.nan)
        steps_timing["Stacking"] = time.time() - t_start_stack
    
    # 5. Flux Calibrate Stacked Image
    stacked_calibrated_success = False
    stacked_hdu = fits.PrimaryHDU(data=stacked_data, header=processed_hdus[0].header.copy())
    
    if cal_stacked_flux:
        try:
            if catalog_stars_df is not None and not catalog_stars_df.empty:
                stacked_hdu, success = aplc.calibrate_flux(stacked_hdu, catalog_stars_df)
                if success:
                    stacked_calibrated_success = True
            else:
                raise ValueError("No catalog stars provided for stacked calibration.")
        except Exception:
            pass
            
    os.makedirs(output_path, exist_ok=True)
    log_file_path = os.path.join(output_path, "stacking_pipeline.log")
    
    base_name, ext = os.path.splitext(output_filename)
    if not ext:
        ext = ".fits"
        
    final_filename = base_name
    if cal_frames_flux and frame_calibrated_success:
        final_filename += "_framecal_flux"
    elif cal_stacked_flux and stacked_calibrated_success:
        final_filename += "_flux"
    else:
        final_filename += "_counts"
    final_filename += ext
    
    dest_fits_path = os.path.join(output_path, final_filename)
    
    stacked_hdul = fits.HDUList([stacked_hdu])
    stacked_hdul.writeto(dest_fits_path, overwrite=True, output_verify="ignore")
    stacked_hdul.close()
    
    preview_png_filename = final_filename.replace(ext, ".png")
    preview_png_path = os.path.join(output_path, preview_png_filename)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(stacked_data, origin='lower', cmap='bone')
    plt.colorbar(label='Intensity')
    plt.title(f"Stacked Image ({method})")
    plt.savefig(preview_png_path, bbox_inches='tight')
    plt.close()
    
    stats_png_path = None
    if output_stats:
        stats_png_filename = final_filename.replace(ext, "_stats.png")
        stats_png_path = os.path.join(output_path, stats_png_filename)
        aplm.simple_hist(
            stacked_data,
            save_path=stats_png_path,
            title=f"Pixel Value Distribution ({method})"
        )
        plt.close()
        
    log_lines = []
    log_lines.append("=== FITS Image Stacking Pipeline ===")
    log_lines.append(f"Time of Initial Request: {time.ctime(start_time)}")
    log_lines.append(f"Stacking Method: {method}")
    log_lines.append(f"Sigma Clipping: {sigma_clip}")
    log_lines.append(f"Background Subtraction: {bg_sub_method}")
    log_lines.append(f"Final Calibration: {final_cal}")
    log_lines.append(f"Calibrate Stacked Flux: {cal_stacked_flux} (Success: {stacked_calibrated_success})")
    log_lines.append(f"Calibrate Frames Flux: {cal_frames_flux} (Success: {frame_calibrated_success})")
    log_lines.append(f"Result Location: {dest_fits_path}")
    log_lines.append(f"Preview Location: {preview_png_path}")
    if stats_png_path:
        log_lines.append(f"Stats Location: {stats_png_path}")
    log_lines.append("\nInput Frames:")
    for path in accepted_paths:
        log_lines.append(f"  - {path}")
        
    if rejected_paths:
        log_lines.append("\nRejected Non-Gaussian Frames:")
        for path in rejected_paths:
            log_lines.append(f"  - {path}")
        
    log_lines.append("\nStep Duration (seconds):")
    for step, duration in steps_timing.items():
        log_lines.append(f"  - {step}: {duration:.4f}s")
        
    if cal_stacked_flux and not stacked_calibrated_success:
        warning_msg = "WARNING: Stacked image flux calibration requested but failed. Saved as sensor counts."
        print(warning_msg)
        log_lines.append(f"\n{warning_msg}")
    elif not cal_stacked_flux:
        warning_msg = "WARNING: Stacked image was not calibrated to flux units. Saved as sensor counts."
        print(warning_msg)
        log_lines.append(f"\n{warning_msg}")
        
    if cal_frames_flux and frame_calibrated_success:
        warning_msg = "WARNING: Each frame was calibrated to flux units before stacking. Output filename reflects this."
        print(warning_msg)
        log_lines.append(f"\n{warning_msg}")
    elif cal_frames_flux and not frame_calibrated_success:
        warning_msg = "WARNING: Individual frame flux calibration requested but failed. Stacking processed on sensor counts."
        print(warning_msg)
        log_lines.append(f"\n{warning_msg}")
        
    log_lines.append(f"\nTime of Completion: {time.ctime()}")
    
    with open(log_file_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
        
    return accepted_paths, dest_fits_path, final_filename


def stack_single_object(
    study_df,
    ra,
    dec,
    crop_size=75,
    catalog="2MASS",
    method="catalog",
    StackingMethod="median",
    sigma_clip=True,
    bg_sub_method="Wavelet Decomposition",
    cal_stacked_flux=True,
    cal_frames_flux=False,
    output_path="./fits/",
    output_filename="stacked_object.fits",
    catalog_stars_df=None,
    **kwargs
):
    """
    Accomplish studying a single object by performing correction, rectification, 
    calibration, cropping, and stacking.
    
    Parameters:
    -----------
    study_df : pandas.DataFrame
        DataFrame representing the observations (from PipeStudy.find_instcals()).
    ra : float
        Right Ascension of the target object in degrees.
    dec : float
        Declination of the target object in degrees.
    crop_size : int or tuple, default 75
        Size of the crop in pixels.
    catalog : str, default "2MASS"
        Catalog to use for calibration and rectification.
    method : str, default "catalog"
        Method for rectification.
    StackingMethod : str, default "median"
        Stacking method ('median' or 'mean').
    sigma_clip : bool or float, default True
        Threshold or flag for sigma clipping.
    bg_sub_method : str, default "Wavelet Decomposition"
        Background subtraction method.
    cal_stacked_flux : bool, default True
        Calibrate final stacked image.
    cal_frames_flux : bool, default False
        Calibrate individual frames before stacking.
    output_path : str, default "./fits/"
        Output path directory.
    output_filename : str, default "stacked_object.fits"
        Output filename.
    catalog_stars_df : pandas.DataFrame, optional
        Pre-queried catalog stars. If None, queries them.
    """
    from package.src.astropipeline import astropipeline_manager as aplmgr
    from package.src.astropipeline import astropipeline_etl as aple

    # 1. Run the manager pipeline to process and crop the images
    cropped_paths = aplmgr.study_single_object(
        study_df=study_df,
        ra=ra,
        dec=dec,
        crop_size=crop_size,
        catalog=catalog,
        method=method,
        output_folder=output_path
    )
    
    if not cropped_paths:
        raise ValueError("No cropped frames were generated.")
        
    # 2. Get and filter catalog stars to the crop footprint
    # Open one cropped frame to read WCS
    hdul = fits.open(cropped_paths[0])
    crop_hdu = None
    for hdu in hdul:
        if isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
            crop_hdu = hdu
            break
    if crop_hdu is None:
        crop_hdu = hdul[0]
    crop_wcs = WCS(crop_hdu.header)
    crop_shape = crop_hdu.data.shape
    hdul.close()

    if catalog_stars_df is None:
        try:
            stars_df = aple.get_catalog_stars(study_df.iloc[0], catalog=catalog)
            valid_stars = []
            for _, star in stars_df.iterrows():
                coord = SkyCoord(star['ra'], star['dec'], unit='deg')
                try:
                    px, py = crop_wcs.world_to_pixel(coord)
                    if 0 <= px < crop_shape[1] and 0 <= py < crop_shape[0]:
                        valid_stars.append(star)
                except Exception:
                    continue
            if valid_stars:
                catalog_stars_df = pd.DataFrame(valid_stars)
            else:
                catalog_stars_df = pd.DataFrame()
        except Exception as e:
            print(f"Failed to query and filter catalog stars: {e}")
            catalog_stars_df = pd.DataFrame()
    else:
        # Filter provided catalog stars to crop footprint
        valid_stars = []
        for _, star in catalog_stars_df.iterrows():
            coord = SkyCoord(star['ra'], star['dec'], unit='deg')
            try:
                px, py = crop_wcs.world_to_pixel(coord)
                if 0 <= px < crop_shape[1] and 0 <= py < crop_shape[0]:
                    valid_stars.append(star)
            except Exception:
                continue
        if valid_stars:
            catalog_stars_df = pd.DataFrame(valid_stars)
        else:
            catalog_stars_df = pd.DataFrame()

    # 3. Stack the cropped images
    return stack_images(
        image_paths=cropped_paths,
        method=StackingMethod,
        sigma_clip=sigma_clip,
        bg_sub_method=bg_sub_method,
        cal_stacked_flux=cal_stacked_flux,
        cal_frames_flux=cal_frames_flux,
        catalog_stars_df=catalog_stars_df,
        output_path=output_path,
        output_filename=output_filename,
        **kwargs
    )
