import sys
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

# Set paths
project_root = r"c:\Users\flann\OneDrive\Documents\gits\doverAstroPipeline"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from package.src.astropipeline import astropipeline_etl as aple
from package.src.astropipeline import astropipeline_correct as aplc
from package.src.astropipeline import astropipeline_manager as aplmgr
from package.src.astropipeline import astropipeline_stack as apls
from package.src.astropipeline import astropipeline_measure as aplm

def run():
    print("--- 1. Cleaning up existing files in fits/ folder ---")
    fits_dir = os.path.join(project_root, "fits")
    
    # Files to backup
    files_to_backup = [
        "kp1852990_dover_rectified.fits",
        "kp1853387_dover_rectified.fits",
        "kp1853391_dover_rectified.fits"
    ]
    for filename in files_to_backup:
        filepath = os.path.join(fits_dir, filename)
        if os.path.exists(filepath):
            bak_path = filepath + ".bak"
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(filepath, bak_path)
            print(f"Backed up {filename} to {filename}.bak")

    # Configure output paths relative to current script/notebook
    aplmgr.output_folder = os.path.join(project_root, "fits") + "/"
    aplmgr.study_output_path = os.path.join(project_root, "fits", "apl_study_dover.csv")

    print("--- 2. Retrieve raw frames ---")
    study = aple.PipeStudy(telescope="kp4m", instrument="newfirm", exposure=10, filter="KXs", max_returns=20)
    df = study.find_instcals()
    df_st = df[(df["OBJECT"] == "ST9107") & (df["caldat"] == "2013-11-06")].reset_index(drop=True)
    print(f"Found {len(df_st)} frames of ST9107.")
    if len(df_st) == 0:
        print("Error: No frames found.")
        return

    print("--- 3. Reprocess frames (correct_subpipe) ---")
    # This will pull/download raw FITS files from scratch
    study_df_corrected = aplmgr.correct_subpipe(df_st.copy())

    print("--- 4. Rectify frames with catalog-based alignment ---")
    study_df_rectified = aplmgr.undistort_subpipe(study_df_corrected.copy(), method="catalog", catalog="2MASS")
    rectified_paths = study_df_rectified["rectified_path"].tolist()
    print("Rectified paths:", rectified_paths)

    print("--- 5. Query catalog & select 10 stars ---")
    row = study_df_rectified.iloc[0]
    stars_df = aple.get_catalog_stars(row, catalog="2MASS")
    stars_df = stars_df.dropna(subset=["Kmag"]).sort_values("Kmag")
    
    # Open reference rectified image (chip 1) to check which catalog stars are in footprint
    hdul_ref = fits.open(rectified_paths[0])
    ref_hdu = None
    for hdu in hdul_ref:
        if isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
            ref_hdu = hdu
            break
    if ref_hdu is None:
        ref_hdu = hdul_ref[0]
    wcs_ref = WCS(ref_hdu.header)
    h_shape = ref_hdu.data.shape
    hdul_ref.close()

    valid_stars = []
    for _, star in stars_df.iterrows():
        c = SkyCoord(star['ra'], star['dec'], unit='deg')
        try:
            px, py = wcs_ref.world_to_pixel(c)
            # Check if star is inside the chip boundaries with a 50-pixel margin
            if 50 <= px <= h_shape[1] - 50 and 50 <= py <= h_shape[0] - 50:
                valid_stars.append(star)
        except Exception:
            continue
            
    valid_stars_df = pd.DataFrame(valid_stars)
    print(f"Total stars queried: {len(stars_df)}. Stars inside stacked chip 1 area: {len(valid_stars_df)}")
    
    # select 10 stars spanning the range of Kmag
    indices = np.linspace(0, len(valid_stars_df) - 1, 10, dtype=int)
    selected_stars = valid_stars_df.iloc[indices].reset_index(drop=True)
    print("Selected 10 stars:")
    print(selected_stars[["ra", "dec", "Kmag", "2MASS"]])

    print("--- 6. Stacking under different settings ---")
    runs = {
        "Run_A_Post_Cal_Wavelet": {"bg_sub_method": "Wavelet Decomposition", "cal_stacked_flux": True, "cal_frames_flux": False, "sigma_clip": True, "filename": "stacked_run_a_demo.fits"},
        "Run_B_Pre_Cal_Wavelet": {"bg_sub_method": "Wavelet Decomposition", "cal_stacked_flux": False, "cal_frames_flux": True, "sigma_clip": True, "filename": "stacked_run_b_demo.fits"},
        "Run_C_Post_Cal_Poly": {"bg_sub_method": "Polynomial 7D", "cal_stacked_flux": True, "cal_frames_flux": False, "sigma_clip": True, "filename": "stacked_run_c_demo.fits"},
        "Run_D_Post_Cal_No_Clip": {"bg_sub_method": "Wavelet Decomposition", "cal_stacked_flux": True, "cal_frames_flux": False, "sigma_clip": False, "filename": "stacked_run_d_demo.fits"}
    }

    run_results = {}
    for name, config in runs.items():
        print(f"Stacking run {name}...")
        _, dest_path, _ = apls.stack_images(
            image_paths=rectified_paths,
            method="median",
            sigma_clip=config["sigma_clip"],
            bg_sub_method=config["bg_sub_method"],
            cal_stacked_flux=config["cal_stacked_flux"],
            cal_frames_flux=config["cal_frames_flux"],
            catalog_stars_df=selected_stars,
            output_path=os.path.join(project_root, "fits") + "/",
            output_filename=config["filename"]
        )
        run_results[name] = dest_path
    print("Finished Stacking runs.")

    print("--- 7. Measure SNR for celestial objects ---")
    def measure_star_snr(data, wcs, ra, dec, best_lvl=1):
        coord = SkyCoord(ra, dec, unit="deg")
        try:
            cutout = Cutout2D(data, coord, (30, 30), wcs=wcs, mode='trim')
            crop_data = cutout.data
        except Exception as e:
            return np.nan
            
        crop_data = np.nan_to_num(crop_data, nan=0.0)
        crop_data[crop_data < 0] = 0.0
        
        crop_dog = aplm.dog_2d(crop_data, sigma_hi=best_lvl, sigma_lo=best_lvl+1, mode='reflect')
        
        center_y, center_x = crop_data.shape[0] // 2, crop_data.shape[1] // 2
        adj_y, adj_x = aplm.adjust_guess_location(crop_data, center_y, center_x, [5, 5])
        
        try:
            sub_crop = aplm.get_adjacent_pixels(crop_data, (adj_y, adj_x), extent=[5, 5], remove_mid=False)
            border_stats = aplm.get_border_stats(sub_crop)
            peak_val = crop_data[adj_y, adj_x]
            
            if border_stats[1] > 0:
                snr = (peak_val - border_stats[0]) / border_stats[1]
            else:
                snr = 0.0
            return snr
        except Exception:
            return np.nan

    # Identify the actual file paths generated (depending on if calibration succeeded or not)
    snr_data = {
        "Star_ID": [f"Star_{i+1}" for i in range(10)],
        "2MASS": selected_stars["2MASS"].tolist(),
        "Kmag": selected_stars["Kmag"].tolist()
    }
    for name, path in run_results.items():
        print(f"Reading and measuring SNR for {name} ({path})...")
        hdul = fits.open(path)
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
        hdul.close()
        
        snrs = []
        for _, star in selected_stars.iterrows():
            snr = measure_star_snr(data, wcs, star["ra"], star["dec"])
            snrs.append(snr)
        snr_data[name] = snrs

    snr_df = pd.DataFrame(snr_data)
    print("SNR Summary Table:")
    print(snr_df)

    print("--- 8. Generating comparison plots ---")
    notebooks_dir = os.path.join(project_root, "package", "dev_notebooks")
    
    star_ids = snr_df["Star_ID"]
    x = np.arange(len(star_ids))
    width = 0.35

    # Determine columns for plots (can be _flux or _counts)
    col_a = [c for c in snr_df.columns if "Run_A_Post_Cal_Wavelet" in c][0]
    col_b = [c for c in snr_df.columns if "Run_B_Pre_Cal_Wavelet" in c][0]
    col_c = [c for c in snr_df.columns if "Run_C_Post_Cal_Poly" in c][0]
    col_d = [c for c in snr_df.columns if "Run_D_Post_Cal_No_Clip" in c][0]

    # Plot 1: Pre vs Post-stack Calibration
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, snr_df[col_b], width, label="Pre-stack Cal", color="orange")
    plt.bar(x + width/2, snr_df[col_a], width, label="Post-stack Cal", color="blue")
    avg_line = (snr_df[col_b] + snr_df[col_a]) / 2
    plt.plot(x, avg_line, color="red", marker="o", label="Average SNR", linewidth=2)
    plt.xticks(x, star_ids)
    plt.xlabel("Celestial Objects")
    plt.ylabel("SNR")
    plt.title("Pre vs Post-stack Calibration SNR Comparison")
    plt.legend()
    plot1_path = os.path.join(notebooks_dir, "plot_pre_vs_post_cal.png")
    plt.savefig(plot1_path, bbox_inches='tight')
    plt.close()
    print("Saved plot to:", plot1_path)

    # Plot 2: Wavelet vs Polynomial background subtraction
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, snr_df[col_a], width, label="Wavelet bg", color="teal")
    plt.bar(x + width/2, snr_df[col_c], width, label="Polynomial 7D bg", color="purple")
    avg_line_bg = (snr_df[col_a] + snr_df[col_c]) / 2
    plt.plot(x, avg_line_bg, color="red", marker="o", label="Average SNR", linewidth=2)
    plt.xticks(x, star_ids)
    plt.xlabel("Celestial Objects")
    plt.ylabel("SNR")
    plt.title("Wavelet vs Polynomial 7D Background Subtraction SNR Comparison")
    plt.legend()
    plot2_path = os.path.join(notebooks_dir, "plot_wavelet_vs_poly.png")
    plt.savefig(plot2_path, bbox_inches='tight')
    plt.close()
    print("Saved plot to:", plot2_path)

    print("--- 9. Save Best Stacked Image ---")
    avg_snrs = {
        "Run_A_Post_Cal_Wavelet": snr_df[col_a].mean(),
        "Run_B_Pre_Cal_Wavelet": snr_df[col_b].mean(),
        "Run_C_Post_Cal_Poly": snr_df[col_c].mean(),
        "Run_D_Post_Cal_No_Clip": snr_df[col_d].mean()
    }
    best_key = max(avg_snrs, key=avg_snrs.get)
    print(f"Best Stacked Image is from: {best_key} with average SNR: {avg_snrs[best_key]:.4f}")
    
    best_stacked_source = run_results[best_key]
    best_stacked_dest = os.path.join(notebooks_dir, "best_stacked_image.fits")
    
    # Load best stacked data to modify header and save it
    hdul = fits.open(best_stacked_source)
    best_data = hdul[0].data
    best_header = hdul[0].header.copy()
    hdul.close()
    
    # Update header descriptively
    best_header["STACKMETHOD"] = ("median", "Image stacking method")
    orig_key = best_key.replace("_demo", "")
    best_header["SIGMACLIP"] = (runs[orig_key]["sigma_clip"], "Sigma clipping threshold applied")
    best_header["BGSUB"] = (runs[orig_key]["bg_sub_method"], "Background subtraction method")
    best_header["CALFRAME"] = (runs[orig_key]["cal_frames_flux"], "Calibration on individual frames")
    best_header["CALSTACK"] = (runs[orig_key]["cal_stacked_flux"], "Calibration on final stacked image")
    best_header["BESTRUN"] = (best_key, "Run name identifier")
    
    avg_snr_val = avg_snrs[best_key]
    if np.isnan(avg_snr_val):
        avg_snr_val = 0.0
    best_header["AVG_SNR"] = (float(avg_snr_val), "Average SNR of selected catalog stars")
    
    best_hdu = fits.PrimaryHDU(data=best_data, header=best_header)
    best_hdul = fits.HDUList([best_hdu])
    best_hdul.writeto(best_stacked_dest, overwrite=True, output_verify="ignore")
    best_hdul.close()
    print("Saved best stacked fits to:", best_stacked_dest)

    print("--- 10. Generate top 3 SNR target zoomed PNGs (3x3 arcminutes) ---")
    best_run_col = [c for c in snr_df.columns if best_key in c][0]
    best_run_snrs = snr_df[best_run_col].tolist()
    snrs_clean = [0.0 if np.isnan(s) else s for s in best_run_snrs]
    top_indices = np.argsort(snrs_clean)[::-1][:3]
    print("Top 3 target indices in best run:", top_indices)
    
    hdul = fits.open(best_stacked_dest)
    best_data = hdul[0].data
    best_wcs = WCS(hdul[0].header)
    hdul.close()
    
    pixel_scale_deg = np.mean(np.abs(best_wcs.pixel_scale_matrix.diagonal()))
    arcmin_3_in_deg = 3.0 / 60.0
    size_pixels = int(np.round(arcmin_3_in_deg / pixel_scale_deg))
    print(f"Calculated 3x3 arcminutes size in pixels: {size_pixels}x{size_pixels} (pixel scale {pixel_scale_deg*3600:.3f} arcsec/pixel)")
    
    for idx in top_indices:
        star = selected_stars.iloc[idx]
        star_name = star["2MASS"]
        ra_val = star["ra"]
        dec_val = star["dec"]
        star_snr = best_run_snrs[idx]
        
        coord = SkyCoord(ra_val, dec_val, unit="deg")
        try:
            cutout = Cutout2D(best_data, coord, (size_pixels, size_pixels), wcs=best_wcs, mode='trim')
            crop_data = cutout.data
            crop_wcs = cutout.wcs
            
            # Find exact pixel location of target in cutout coordinates
            center_x, center_y = cutout.wcs.world_to_pixel(coord)
            
            plt.figure(figsize=(6, 6))
            vmin = np.nanpercentile(crop_data, 1)
            vmax = np.nanpercentile(crop_data, 99.5)
            plt.imshow(crop_data, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            
            plt.axvline(center_x, color='red', linestyle='--', alpha=0.5)
            plt.axhline(center_y, color='red', linestyle='--', alpha=0.5)
            circle = plt.Circle((center_x, center_y), size_pixels // 20, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            
            plt.colorbar(label='Flux (Jy)' if best_header.get("BUNIT") == "Jy" else 'Counts')
            plt.title(f"Target: 2MASS J{star_name}\nSNR: {star_snr:.2f}, Kmag: {star['Kmag']:.2f}")
            
            out_png_name = f"zoom_2MASS_J{star_name}.png"
            out_png_path = os.path.join(notebooks_dir, out_png_name)
            plt.savefig(out_png_path, bbox_inches='tight')
            plt.close()
            print(f"Saved zoomed target to: {out_png_path}")
        except Exception as e:
            print(f"Failed to generate zoom for star {star_name}: {str(e)}")

if __name__ == "__main__":
    run()
