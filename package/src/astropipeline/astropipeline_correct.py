import re
import numpy as np
import pandas as pd
import random
import astropy.io.fits as pyfits
from astropy.nddata import Cutout2D
from package.src.astropipeline.astropipeline_etl import cached_fits_open
from scipy.interpolate import LinearNDInterpolator
import pywt
from astropy.wcs import WCS
import scipy.ndimage as ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from package.src.astropipeline import astropipeline_gpu as apgpu



def load_mask(dqm_mask_url, keep_list=(0)):

    dqm_fits = cached_fits_open(dqm_mask_url)
    crop_ranges = dict()
    pixel_mask = dict()

    for index, hdu in enumerate(dqm_fits):
        if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
            if isinstance(hdu.data, np.ndarray):

                row_range = range(1, np.size(hdu.data, axis=0) + 1)
                col_range = range(1, np.size(hdu.data, axis=1) + 1)

                mask_slice = np.ix_(row_range, col_range)

                pixel_mask[index] = ~np.isin(hdu.data, keep_list)

                crop_ranges[index] = mask_slice

    return pixel_mask, crop_ranges


def get_dark_vals(dark_fits_urls, number_samples, crop_ranges):

    dark_val_arrays = dict()
    dark_val_counters = dict()
    dark_val_means = dict()

    url_indexes = random.choices(
        np.arange(0, len(dark_fits_urls)),
        k=number_samples
        )

    url_list = dark_fits_urls.iloc[url_indexes]

    for url in url_list:
        dark_fits = cached_fits_open(url)

        for index, hdu in enumerate(dark_fits):
            if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
                if isinstance(hdu.data, np.ndarray):
                    if index not in dark_val_arrays:
                        dark_val_arrays[index] = hdu.data[crop_ranges[index]]
                        dark_val_counters[index] = 1
                    else:
                        dark_val_arrays[index] = (
                            dark_val_arrays[index] +
                            hdu.data[crop_ranges[index]]
                        )
                        dark_val_counters[index] += 1

    for index in dark_val_counters:
        dark_val_means[index] = dark_val_arrays[index] /\
                                dark_val_counters[index]

    return dark_val_means, dark_val_counters, url_indexes


def get_gain_vals(
    dark_val_means, flat_fits_urls, number_samples, crop_ranges, pixel_mask
):

    flat_val_arrays = dict()
    flat_val_counters = dict()
    gain_vals = dict()
    flat_val_cumtime = dict()

    url_indexes = random.choices(
        np.arange(0, len(flat_fits_urls)),
        k=number_samples
        )
    url_list = flat_fits_urls.iloc[url_indexes]

    for url in url_list:
        flat_fits = cached_fits_open(url)

        for index, hdu in enumerate(flat_fits):
            if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
                if isinstance(hdu.data, np.ndarray):
                    vals_dark_remove = (
                        hdu.data[crop_ranges[index]] - dark_val_means[index]
                    )
                    if index not in flat_val_arrays:
                        flat_val_arrays[index] = vals_dark_remove
                        flat_val_counters[index] = 1
                        flat_val_cumtime[index] = hdu.header["EXPTIME"]
                    else:
                        flat_val_arrays[index] = (
                            flat_val_arrays[index] + vals_dark_remove
                        )
                        flat_val_counters[index] += 1
                        flat_val_cumtime[index] += hdu.header["EXPTIME"]

    for index in flat_val_counters:
        gain_vals[index] = flat_val_arrays[index] / flat_val_counters[index]
        gain_vals[index][pixel_mask[index]] = np.nan

    return gain_vals, flat_val_counters, flat_val_cumtime, url_indexes


def parse_wat_table(hdu, wat_df=pd.DataFrame()):

    num_axis = hdu.header["NAXIS"]

    for axis in range(1, num_axis + 1):
        wat_bin = ""
        for hdr_line in hdu.header:
            if ("WAT" + str(axis)) in hdr_line:
                wat_bin = wat_bin + hdu.header[hdr_line] + " "

        this_row = len(wat_df)
        wat_df.loc[this_row, "wtype"] = re.search(r"wtype=(\w+)",
                                                  wat_bin).group(1)
        wat_df.loc[this_row, "axtype"] = re.search(r"axtype=(\w+)",
                                                   wat_bin).group(1)
        wat_df.loc[this_row, "dc_vals"] = re.search(
            r'(lngcor|latcor) = "([^"]*)"', wat_bin
        ).group(2)

    return wat_df


def image_uniformity_correct(
    raw_fits_image_url, dark_val_means, gain_vals, crop_ranges
):

    raw_fits = cached_fits_open(raw_fits_image_url)
    temp_fits = raw_fits.copy()
    if "RADECSYS" in temp_fits[0].header:
        temp_fits[0].header["RADESYSa"] = temp_fits[0].header["RADECSYS"]
        temp_fits[0].header.remove("RADECSYS")

    temp_fits.writeto("./fits/temp.fits.fz", overwrite=True)

    with pyfits.open("./fits/temp.fits.fz", update=True) as raw_fits:

        balanced_fits = raw_fits.copy()

        balanced_fits[0].verify("fix+exception")
        balanced_fits[4].verify("fix+exception")

        wat_df = pd.DataFrame()
        for index, hdu in enumerate(balanced_fits):

            if isinstance(hdu.data, np.ndarray):

                if index > 1:
                    wat_df = parse_wat_table(hdu, wat_df)
                else:
                    wat_df = parse_wat_table(hdu)

                try:
                    wcs = WCS(hdu.header)
                except:
                    print(
                        "OH NO ---- BIG PROBLEMS WITH WCS FOR EXTENSION: " +
                        str(index)
                    )
                    balanced_fits.pop(index)
                    continue

                # Update the FITS header with the cutout WCS
                balanced_fits[index].data = (
                    hdu.data[crop_ranges[index]] - dark_val_means[index]
                ) / gain_vals[index]

                balanced_fits[index].data[gain_vals[index] == 0] = 0
                min_row = min(crop_ranges[index][0])[0]
                max_row = max(crop_ranges[index][0])[0]
                min_col = min(crop_ranges[index][1][0])
                max_col = max(crop_ranges[index][1][0])

                position = ((max_row - min_row) / 2, (max_col - min_col) / 2)
                size = (max_row - min_row, max_col - min_col)

                cutout = Cutout2D(
                    hdu.data,
                    wcs.pixel_to_world(position[0], position[1]),
                    size,
                    mode="trim",
                    wcs=wcs,
                )

                balanced_fits[index].data = cutout.data
                balanced_fits[index].header = cutout.wcs.to_header()

        fix_exception = balanced_fits[0].verify(option="fix+exception")

        print(fix_exception)

        balanced_fits[0].update_header()

        return balanced_fits


def heal_pixels(fits_image, method="mean", element_select=(-1)):

    healed_fits = fits_image.copy()

    for index, hdu in enumerate(fits_image):
        if not ((index in element_select) | (-1 in element_select)):
            continue

        is_array = False
        if isinstance(hdu.data, np.ndarray):
            is_array = True
        elif apgpu.HAS_GPU and apgpu.cp is not None and isinstance(hdu.data, apgpu.cp.ndarray):
            is_array = True

        if is_array:
            xp = apgpu.get_array_module(hdu.data)
            dq0_mask = xp.isnan(hdu.data)  # invalid elements
            valid_elements = hdu.data[~dq0_mask]

            if method == "mean":
                if len(valid_elements) > 0:
                    mean_val = xp.mean(valid_elements)
                else:
                    mean_val = 0.0
                hdu.data[dq0_mask] = mean_val
            elif method in ("linear", "quadratic"):
                if method == "quadratic":
                    print("Quadratic interpolation not incorporated yet, using linear local mean fallback")
                
                hdu_data_cpu = apgpu.to_cpu(hdu.data)
                from astropy.convolution import interpolate_replace_nans, Box2DKernel
                kernel = Box2DKernel(5)
                interpolated_data = interpolate_replace_nans(hdu_data_cpu, kernel, boundary="extend")
                
                remaining_nans = np.isnan(interpolated_data)
                if np.any(remaining_nans):
                    if len(valid_elements) > 0:
                        mean_val = apgpu.to_cpu(xp.mean(valid_elements))
                    else:
                        mean_val = 0.0
                    interpolated_data[remaining_nans] = mean_val
                
                healed_fits[index].data = apgpu.to_gpu(interpolated_data) if xp != np else interpolated_data

    return healed_fits


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    std[std == 0] = 1.0
    scale = np.sqrt(2) / std
    T = np.array([
        [scale[0], 0, -mean[0]*scale[0]],
        [0, scale[1], -mean[1]*scale[1]],
        [0, 0, 1]
    ])
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_hom.T).T[:, :2]
    return pts_norm, T


def estimate_homography(pts_src, pts_dst):
    src_norm, T_src = normalize_points(pts_src)
    dst_norm, T_dst = normalize_points(pts_dst)
    A = []
    for i in range(len(src_norm)):
        u, v = src_norm[i]
        x, y = dst_norm[i]
        A.append([-u, -v, -1, 0, 0, 0, x*u, x*v, x])
        A.append([0, 0, 0, -u, -v, -1, y*u, y*v, y])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H


def apply_homography(H, pts):
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_trans_hom = (H @ pts_hom.T).T
    denom = pts_trans_hom[:, 2:3]
    denom[denom == 0] = 1e-10
    pts_trans = pts_trans_hom[:, :2] / denom
    return pts_trans


def ransac_homography(pts_src, pts_dst, threshold=3.0, max_iters=1000):
    best_H = None
    best_inliers = []
    n_points = len(pts_src)
    if n_points < 4:
        return None, []
    for _ in range(max_iters):
        indices = np.random.choice(n_points, 4, replace=False)
        src_sample = pts_src[indices]
        dst_sample = pts_dst[indices]
        try:
            H = estimate_homography(src_sample, dst_sample)
        except Exception:
            continue
        dst_pred = apply_homography(H, pts_src)
        errors = np.linalg.norm(dst_pred - pts_dst, axis=1)
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    if len(best_inliers) >= 4:
        try:
            best_H = estimate_homography(pts_src[best_inliers], pts_dst[best_inliers])
        except Exception:
            pass
    return best_H, best_inliers


def rectify_wcs(hdu, log_func=print):
    if not isinstance(hdu.data, np.ndarray):
        return None
    wcs = WCS(hdu.header)
    if not wcs.is_celestial:
        raise ValueError("WCS is not celestial")
    ny, nx = hdu.data.shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_sky = wcs.pixel_to_world(center_x, center_y)
    
    sky_c = center_sky
    sky_dx = wcs.pixel_to_world(center_x + 1, center_y)
    sky_dy = wcs.pixel_to_world(center_x, center_y + 1)
    cdelt1 = sky_c.separation(sky_dx).deg
    cdelt2 = sky_c.separation(sky_dy).deg
    
    if np.isnan(cdelt1) or np.isnan(cdelt2) or cdelt1 == 0 or cdelt2 == 0:
        from astropy.wcs.utils import proj_plane_pixel_scales
        try:
            scales = proj_plane_pixel_scales(wcs)
            cdelt1, cdelt2 = scales[0], scales[1]
        except Exception:
            cdelt1, cdelt2 = 0.0001, 0.0001
            
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [center_x + 1, center_y + 1]
    target_wcs.wcs.crval = [center_sky.ra.deg, center_sky.dec.deg]
    target_wcs.wcs.cdelt = [-cdelt1, cdelt2]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    target_wcs.wcs.cunit = ["deg", "deg"]
    
    y_grid, x_grid = np.indices(hdu.data.shape)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    
    world_coords = target_wcs.pixel_to_world(x_flat, y_flat)
    x_old, y_old = wcs.world_to_pixel(world_coords)
    
    if apgpu.HAS_GPU:
        try:
            data_gpu = apgpu.to_gpu(hdu.data)
            y_old_gpu = apgpu.to_gpu(y_old.reshape(hdu.data.shape))
            x_old_gpu = apgpu.to_gpu(x_old.reshape(hdu.data.shape))
            cp_ndimage = apgpu.get_ndimage()
            resampled_data_gpu = cp_ndimage.map_coordinates(
                data_gpu,
                [y_old_gpu, x_old_gpu],
                order=1,
                cval=np.nan
            )
            resampled_data = apgpu.to_cpu(resampled_data_gpu)
        except Exception as e:
            log_func(f"GPU map_coordinates in rectify_wcs failed, falling back to CPU: {str(e)}")
            resampled_data = ndimage.map_coordinates(
                hdu.data,
                [y_old.reshape(hdu.data.shape), x_old.reshape(hdu.data.shape)],
                order=1,
                cval=np.nan
            )
    else:
        resampled_data = ndimage.map_coordinates(
            hdu.data,
            [y_old.reshape(hdu.data.shape), x_old.reshape(hdu.data.shape)],
            order=1,
            cval=np.nan
        )
    
    new_header = hdu.header.copy()
    for key in list(new_header.keys()):
        if any(p in key for p in ["CRPIX", "CRVAL", "CDELT", "CTYPE", "CUNIT", "CD1_", "CD2_", "PC1_", "PC2_", "PV1_", "PV2_", "A_", "B_", "AP_", "BP_", "WAT", "CQDIS", "DQ", "DP", "TPD"]):
            if key in new_header:
                del new_header[key]
    new_header.update(target_wcs.to_header())
    
    if isinstance(hdu, pyfits.PrimaryHDU):
        rectified_hdu = pyfits.PrimaryHDU(data=resampled_data, header=new_header)
    else:
        rectified_hdu = pyfits.ImageHDU(data=resampled_data, header=new_header)
    return rectified_hdu


def rectify_catalog(hdu, catalog_stars_df, log_func=print, offset=None):
    if not isinstance(hdu.data, np.ndarray):
        return None
    try:
        wcs = WCS(hdu.header)
    except Exception:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [hdu.data.shape[1]/2, hdu.data.shape[0]/2]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [-0.0001, 0.0001]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
    ny, nx = hdu.data.shape
    center_x = (nx - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_sky = wcs.pixel_to_world(center_x, center_y)
    
    from astropy.wcs.utils import proj_plane_pixel_scales
    try:
        scales = proj_plane_pixel_scales(wcs)
        cdelt1, cdelt2 = scales[0], scales[1]
    except Exception:
        cdelt1, cdelt2 = 0.0001, 0.0001
        
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [center_x + 1, center_y + 1]
    target_wcs.wcs.crval = [center_sky.ra.deg, center_sky.dec.deg]
    target_wcs.wcs.cdelt = [-cdelt1, cdelt2]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    target_wcs.wcs.cunit = ["deg", "deg"]
    
    from package.src.astropipeline import astropipeline_measure as aplm
    from astropy.coordinates import SkyCoord
    
    # Run extract_star_samples on the image to locate candidate stars
    try:
        detected_stars_dict = aplm.extract_star_samples(hdu, extensions=0, num_peaks=200)
        detected_stars = detected_stars_dict.get(0, [])
    except Exception as e:
        log_func(f"Failed to extract star samples using band-pass filtering: {str(e)}")
        detected_stars = []

    pts_rect = []
    pts_img = []
    
    # 1. Map catalog stars to initial pixel coordinates
    catalog_pixels = []
    for _, star in catalog_stars_df.iterrows():
        c = SkyCoord(star['ra'], star['dec'], frame=hdu.header.get('RADESYS', 'ICRS').lower(), unit="deg")
        try:
            x_img_init, y_img_init = wcs.world_to_pixel(c)
            if not np.isnan(x_img_init) and not np.isnan(y_img_init):
                if -2000 <= x_img_init <= nx + 2000 and -2000 <= y_img_init <= ny + 2000:
                    catalog_pixels.append((x_img_init, y_img_init, c))
        except Exception:
            continue
            
    # 2. Find systematic translation offset (Consensus Peak)
    if offset is not None:
        best_dx, best_dy = offset
        log_func(f"Using global pointing offset prior: dx={best_dx}, dy={best_dy} pixels")
    else:
        from collections import Counter
        offsets = []
        for cx, cy, _ in catalog_pixels:
            for ds in detected_stars:
                dx = ds['col'] - cx
                dy = ds['row'] - cy
                # Bin the offsets to nearest 2 pixels to find consensus peak
                offsets.append((round(dx/2.0)*2.0, round(dy/2.0)*2.0))
                
        most_common = Counter(offsets).most_common(1)
        if most_common and most_common[0][1] >= 3:
            best_dx, best_dy = most_common[0][0]
            log_func(f"Detected systematic pointing offset: dx={best_dx}, dy={best_dy} pixels (based on {most_common[0][1]} consensus matches)")
        else:
            best_dx, best_dy = 0.0, 0.0
            log_func("No clear systematic pointing offset detected.")
        
    # 3. Match stars using shifted catalog positions
    for cx, cy, c in catalog_pixels:
        x_shifted = cx + best_dx
        y_shifted = cy + best_dy
        
        # Find the closest detected star to the shifted catalog position
        best_match = None
        min_dist = float('inf')
        for ds in detected_stars:
            dist = np.sqrt((ds['col'] - x_shifted)**2 + (ds['row'] - y_shifted)**2)
            if dist < min_dist:
                min_dist = dist
                best_match = ds
                
        # Match within 15 pixels after correcting for translation
        if best_match is not None and min_dist < 15.0:
            x_img = best_match['col']
            y_img = best_match['row']
            x_rect, y_rect = target_wcs.world_to_pixel(c)
            pts_rect.append([x_rect, y_rect])
            pts_img.append([x_img, y_img])
            
    pts_rect = np.array(pts_rect)
    pts_img = np.array(pts_img)
    if len(pts_rect) < 4:
        raise ValueError(f"Insufficient matched celestial objects ({len(pts_rect)}) to perform RANSAC SolvePnP correction.")
        
    best_H, inliers = ransac_homography(pts_rect, pts_img, threshold=3.0)
    if best_H is None or len(inliers) < 4:
        raise ValueError("RANSAC homography estimation failed.")
        
    num_matched = len(inliers)
    pts_img_pred = apply_homography(best_H, pts_rect[inliers])
    errors_pixels = np.linalg.norm(pts_img_pred - pts_img[inliers], axis=1)
    pixel_scale = np.mean([cdelt1, cdelt2])
    errors_deg = errors_pixels * pixel_scale
    mean_err = np.mean(errors_deg)
    rmse_err = np.sqrt(np.mean(errors_deg**2))
    
    log_func(f"Alignment Metrics:")
    log_func(f"  - Number of matched celestial objects (catalog vs image): {num_matched}")
    log_func(f"  - Mean error distance (angular subtense): {mean_err:.6f} degrees ({mean_err*3600:.3f} arcsec)")
    log_func(f"  - Root-mean-square error (angular subtense): {rmse_err:.6f} degrees ({rmse_err*3600:.3f} arcsec)")
    
    y_grid, x_grid = np.indices(hdu.data.shape)
    pts_rect_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    pts_img_grid = apply_homography(best_H, pts_rect_grid)
    x_old = pts_img_grid[:, 0].reshape(hdu.data.shape)
    y_old = pts_img_grid[:, 1].reshape(hdu.data.shape)
    
    if apgpu.HAS_GPU:
        try:
            data_gpu = apgpu.to_gpu(hdu.data)
            y_old_gpu = apgpu.to_gpu(y_old)
            x_old_gpu = apgpu.to_gpu(x_old)
            cp_ndimage = apgpu.get_ndimage()
            resampled_data_gpu = cp_ndimage.map_coordinates(
                data_gpu,
                [y_old_gpu, x_old_gpu],
                order=1,
                cval=np.nan
            )
            resampled_data = apgpu.to_cpu(resampled_data_gpu)
        except Exception as e:
            log_func(f"GPU map_coordinates in rectify_catalog failed, falling back to CPU: {str(e)}")
            resampled_data = ndimage.map_coordinates(
                hdu.data,
                [y_old, x_old],
                order=1,
                cval=np.nan
            )
    else:
        resampled_data = ndimage.map_coordinates(
            hdu.data,
            [y_old, x_old],
            order=1,
            cval=np.nan
        )
    
    # Region of successful rectification defined by the warped boundaries of the original image
    corners_img = np.array([
        [0, 0],
        [nx - 1, 0],
        [nx - 1, ny - 1],
        [0, ny - 1]
    ], dtype=float)
    try:
        H_inv = np.linalg.inv(best_H)
        corners_rect = apply_homography(H_inv, corners_img)
    except Exception:
        corners_rect = pts_rect[inliers]
        
    hull = ConvexHull(corners_rect)
    hull_path = Path(corners_rect[hull.vertices])
    mask_inside = hull_path.contains_points(pts_rect_grid).reshape(hdu.data.shape)
    resampled_data[~mask_inside] = np.nan
    
    new_header = hdu.header.copy()
    for key in list(new_header.keys()):
        if any(p in key for p in ["CRPIX", "CRVAL", "CDELT", "CTYPE", "CUNIT", "CD1_", "CD2_", "PC1_", "PC2_", "PV1_", "PV2_", "A_", "B_", "AP_", "BP_", "WAT", "CQDIS", "DQ", "DP", "TPD"]):
            if key in new_header:
                del new_header[key]
    new_header.update(target_wcs.to_header())
    new_header["MATCHED"] = num_matched
    new_header["MEANERR"] = mean_err
    new_header["RMSEERR"] = rmse_err
    
    if isinstance(hdu, pyfits.PrimaryHDU):
        rectified_hdu = pyfits.PrimaryHDU(data=resampled_data, header=new_header)
    else:
        rectified_hdu = pyfits.ImageHDU(data=resampled_data, header=new_header)
    return rectified_hdu


def calculate_global_pointing_offset(fits_in, catalog_stars_df, log_func=print):
    """
    Calculate the global systematic pointing offset (best_dx, best_dy) in pixels
    by aggregating translation offsets across all extensions of a FITS file.
    """
    from collections import Counter
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from package.src.astropipeline import astropipeline_measure as aplm
    import numpy as np

    offsets = []
    log_func("Calculating global consensus pointing offset across all extensions...")

    for index, hdu in enumerate(fits_in):
        if not isinstance(hdu.data, np.ndarray) or hdu.data.ndim != 2:
            continue

        try:
            wcs = WCS(hdu.header)
            if not wcs.is_celestial:
                continue
        except Exception:
            continue

        ny, nx = hdu.data.shape

        # Run extract_star_samples to find candidate stars
        try:
            detected_stars_dict = aplm.extract_star_samples(hdu, extensions=0, num_peaks=200)
            detected_stars = detected_stars_dict.get(0, [])
        except Exception as e:
            log_func(f"  Extension {index}: Failed to extract star samples: {str(e)}")
            continue

        if not detected_stars:
            continue

        # Project catalog stars to this extension's nominal pixel coordinates
        catalog_pixels = []
        for _, star in catalog_stars_df.iterrows():
            c = SkyCoord(star['ra'], star['dec'], frame=hdu.header.get('RADESYS', 'ICRS').lower(), unit="deg")
            try:
                x_img_init, y_img_init = wcs.world_to_pixel(c)
                if not np.isnan(x_img_init) and not np.isnan(y_img_init):
                    # Buffer of 2000 pixels
                    if -2000 <= x_img_init <= nx + 2000 and -2000 <= y_img_init <= ny + 2000:
                        catalog_pixels.append((x_img_init, y_img_init))
            except Exception:
                continue

        # Accumulate offsets
        for cx, cy in catalog_pixels:
            for ds in detected_stars:
                dx = ds['col'] - cx
                dy = ds['row'] - cy
                offsets.append((round(dx / 2.0) * 2.0, round(dy / 2.0) * 2.0))

    most_common = Counter(offsets).most_common(1)
    if most_common and most_common[0][1] >= 3:
        best_dx, best_dy = most_common[0][0]
        log_func(f"Global pointing offset detected: dx={best_dx}, dy={best_dy} pixels (based on {most_common[0][1]} consensus matches)")
        return best_dx, best_dy
    else:
        log_func("No clear global systematic pointing offset detected.")
        return 0.0, 0.0


def rectify_image(hdu, method="wcs", catalog_stars_df=None, log_func=print, offset=None):
    if method == "wcs":
        try:
            wcs = WCS(hdu.header)
            if not wcs.is_celestial:
                raise ValueError("WCS is not celestial")
            test_world = wcs.pixel_to_world(0, 0)
            if np.isnan(test_world.ra.deg) or np.isnan(test_world.dec.deg):
                raise ValueError("WCS maps to NaN")
            log_func("Attempting rectilinear correction based on WCS header data.")
            return rectify_wcs(hdu, log_func=log_func)
        except Exception as e:
            log_func(f"WARNING: WCS header is corrupted or invalid ({str(e)}). Switching to star catalog correction.")
            method = "catalog"
    if method == "catalog":
        if catalog_stars_df is None or len(catalog_stars_df) == 0:
            raise ValueError("No catalog stars provided or available for catalog-based rectification.")
        log_func("Attempting rectilinear correction based on star catalog data.")
        return rectify_catalog(hdu, catalog_stars_df, log_func=log_func, offset=offset)
    raise ValueError(f"Unknown rectification method: {method}")


def fit_poly_2d(data, degree=7):
    if apgpu.HAS_GPU:
        try:
            cp = apgpu.cp
            data_gpu = apgpu.to_gpu(data)
            ny, nx = data_gpu.shape
            y, x = cp.indices((ny, nx))
            mask = ~cp.isnan(data_gpu)
            x_valid = x[mask]
            y_valid = y[mask]
            z_valid = data_gpu[mask]
            
            n_terms = (degree + 1) * (degree + 2) // 2
            if len(z_valid) < n_terms:
                degree = 0
                
            A = []
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    A.append((x_valid ** i) * (y_valid ** j))
            A = cp.column_stack(A)
            
            coeff, _, _, _ = cp.linalg.lstsq(A, z_valid, rcond=None)
            
            background = cp.zeros((ny, nx))
            idx = 0
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    background += coeff[idx] * (x ** i) * (y ** j)
                    idx += 1
            return apgpu.to_cpu(background)
        except Exception as e:
            print(f"GPU fit_poly_2d failed, falling back to CPU: {str(e)}")

    ny, nx = data.shape
    y, x = np.indices((ny, nx))
    mask = ~np.isnan(data)
    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = data[mask]
    
    n_terms = (degree + 1) * (degree + 2) // 2
    if len(z_valid) < n_terms:
        degree = 0
        
    A = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            A.append((x_valid ** i) * (y_valid ** j))
    A = np.column_stack(A)
    
    coeff, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    
    background = np.zeros((ny, nx))
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            background += coeff[idx] * (x ** i) * (y ** j)
            idx += 1
    return background


def fit_wavelet_background(data, wavelet='db2', level=4):
    clean_data = data.copy()
    nan_mask = np.isnan(clean_data)
    if np.any(nan_mask):
        median_val = np.nanmedian(clean_data)
        if np.isnan(median_val):
            median_val = 0.0
        clean_data[nan_mask] = median_val
        
    max_level = pywt.dwtn_max_level(clean_data.shape, wavelet)
    actual_level = min(level, max_level)
    
    if actual_level <= 0:
        return np.zeros_like(data)
        
    coeffs = pywt.wavedec2(clean_data, wavelet=wavelet, level=actual_level)
    for i in range(1, len(coeffs)):
        coeffs[i] = tuple([np.zeros_like(c) for c in coeffs[i]])
        
    background = pywt.waverec2(coeffs, wavelet=wavelet)
    background = background[:data.shape[0], :data.shape[1]]
    return background


def subtract_background(data, method="None"):
    if method == "None" or method is None:
        return data.copy()
    elif method == "Linear":
        bg = fit_poly_2d(data, degree=1)
        return data - bg
    elif method == "Polynomial 7D":
        bg = fit_poly_2d(data, degree=7)
        return data - bg
    elif method == "Wavelet Decomposition":
        bg = fit_wavelet_background(data, wavelet='db2', level=4)
        return data - bg
    else:
        raise ValueError(f"Unknown background subtraction method: {method}")


def calibrate_flux(hdu, catalog_stars_df, log_func=print):
    if not isinstance(hdu.data, np.ndarray):
        return hdu, False
        
    try:
        wcs = WCS(hdu.header)
    except Exception as e:
        raise ValueError(f"Cannot perform flux calibration without a valid WCS header: {str(e)}")
        
    from astropy.coordinates import SkyCoord
    from package.src.astropipeline import astropipeline_measure as aplm
    
    matched_zp = []
    gain = hdu.header.get("GAIN", hdu.header.get("DETGAIN", 1.0))
    
    for _, star in catalog_stars_df.iterrows():
        c = SkyCoord(star['ra'], star['dec'], frame=hdu.header.get('RADESYS', 'ICRS').lower(), unit="deg")
        try:
            px_val, py_val = wcs.world_to_pixel(c)
            if np.isnan(px_val) or np.isnan(py_val):
                continue
            if not (0 <= px_val < hdu.data.shape[1] and 0 <= py_val < hdu.data.shape[0]):
                continue
        except Exception:
            continue
            
        try:
            cutout = Cutout2D(hdu.data, c, (100, 100), mode='trim', wcs=wcs)
        except Exception:
            continue
            
        try:
            found_coords, _ = aplm.wdec_bandpass_find(
                image=cutout.data,
                num_returns=3,
                wavelet='db2',
                start_level=0,
                stop_level=3
            )
        except Exception:
            continue
            
        peaks = []
        for lvl_data in found_coords:
            row_coords = lvl_data[2][0]
            col_coords = lvl_data[2][1]
            for r, c_ in zip(row_coords, col_coords):
                peaks.append((r, c_))
                
        if not peaks:
            continue
            
        cy_cut, cx_cut = cutout.data.shape[0]/2.0, cutout.data.shape[1]/2.0
        best_peak = None
        min_dist = float('inf')
        for r, c_ in peaks:
            dist = np.sqrt((r - cy_cut)**2 + (c_ - cx_cut)**2)
            if dist < min_dist:
                min_dist = dist
                best_peak = (r, c_)
                
        if best_peak is not None and min_dist < 20.0:
            r_peak, c_peak = best_peak
            r_start = max(0, r_peak - 4)
            r_end = min(cutout.data.shape[0], r_peak + 5)
            c_start = max(0, c_peak - 4)
            c_end = min(cutout.data.shape[1], c_peak + 5)
            sub_patch = cutout.data[r_start:r_end, c_start:c_end]
            
            bg = np.median(sub_patch) if sub_patch.size > 0 else 0.0
            
            yy, xx = np.indices(sub_patch.shape)
            y_center, x_center = r_peak - r_start, c_peak - c_start
            dist_mask = (yy - y_center)**2 + (xx - x_center)**2 <= 4**2
            
            star_counts = np.sum(sub_patch[dist_mask] - bg)
            if star_counts > 0:
                inst_mag = -2.5 * np.log10(star_counts)
                cat_mag = star.get('mag', star.get('Kmag', star.get('dec', 10.0)))
                try:
                    cat_mag = float(cat_mag)
                except ValueError:
                    cat_mag = 10.0
                zp = cat_mag - inst_mag
                matched_zp.append(zp)
                
    if len(matched_zp) < 3:
        log_func(f"WARNING: Insufficient catalog stars matched ({len(matched_zp)}) for reliable flux calibration.")
        return hdu, False
        
    matched_zp = np.array(matched_zp)
    filtered_zp = matched_zp.copy()
    iterations = 0
    max_iterations = 5
    while iterations < max_iterations:
        if len(filtered_zp) < 3:
            break
        mean = np.mean(filtered_zp)
        std = np.std(filtered_zp, ddof=1) if len(filtered_zp) > 1 else 0.0
        if std == 0.0:
            break
        mask = np.abs(filtered_zp - mean) <= 2.0 * std
        new_filtered = filtered_zp[mask]
        if len(new_filtered) == len(filtered_zp):
            break
        filtered_zp = new_filtered
        iterations += 1
        
    if len(filtered_zp) < 3:
        log_func(f"WARNING: Insufficient catalog stars matched ({len(filtered_zp)}) after outlier rejection for reliable flux calibration.")
        return hdu, False
        
    zp_mean = np.mean(filtered_zp)
    zp_var = np.var(filtered_zp)
    
    var_zp = zp_var / len(filtered_zp) if len(filtered_zp) > 1 else 0.0
    var_catalog = zp_var
    var_gain = 0.01
    var_combined = var_zp + var_gain + var_catalog
    
    f0 = 3631.0
    scale_factor = f0 * (10 ** (-0.4 * zp_mean))
    
    calibrated_data = hdu.data * scale_factor
    
    new_header = hdu.header.copy()
    new_header["BUNIT"] = "Jy"
    new_header["PHOTZP"] = (zp_mean, "Photometric zero point")
    new_header["PHOTZPV"] = (var_zp, "Photometric zero point variance")
    new_header["DETGAIN"] = (gain, "Detector gain")
    new_header["CATVAR"] = (var_catalog, "Catalog matching variance")
    new_header["CALUNC"] = (np.sqrt(var_combined), "Combined calibration uncertainty (fractional)")
    new_header["PHOTSYS"] = ("AB", "Photometric system")
    new_header["PHOTPLAM"] = (2.2, "Pivot wavelength (microns)")
    
    if isinstance(hdu, pyfits.PrimaryHDU):
        cal_hdu = pyfits.PrimaryHDU(data=calibrated_data, header=new_header)
    else:
        cal_hdu = pyfits.ImageHDU(data=calibrated_data, header=new_header)
        
    return cal_hdu, True


