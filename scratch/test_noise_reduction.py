import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

project_root = r"c:\Users\flann\OneDrive\Documents\gits\doverAstroPipeline"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from package.src.astropipeline import astropipeline_etl as aple
from package.src.astropipeline import astropipeline_manager as aplmgr
from package.src.astropipeline import astropipeline_stack as apls

def main():
    print("Finding ST9107 observations from 2013-11-06...")
    study = aple.PipeStudy(telescope="kp4m", instrument="newfirm", exposure=10, filter="KXs", max_returns=20)
    df = study.find_instcals()
    df_st = df[(df["OBJECT"] == "ST9107") & (df["caldat"] == "2013-11-06")].reset_index(drop=True)
    print(f"Found {len(df_st)} frames.")
    
    # Run the manager correction, rectification, and calibration on the first 5 frames for speed
    test_df = df_st.head(5).copy()
    
    # Set manager paths
    aplmgr.output_folder = os.path.join(project_root, "fits") + "/"
    aplmgr.study_output_path = os.path.join(project_root, "fits", "apl_study_dover.csv")
    
    # Sexagesimal coordinates of ST9107
    coord = SkyCoord('3:32:37.77', '37:27:23.7', unit=(u.hourangle, u.deg))
    ra, dec = coord.ra.deg, coord.dec.deg
    
    print(f"Running stack_single_object for ST9107 (RA: {ra:.5f}, Dec: {dec:.5f}) with 1 to 5 frames...")
    
    noises = []
    for n in range(1, 6):
        subset_df = test_df.head(n).copy()
        
        accepted_paths, dest_fits_path, final_filename = apls.stack_single_object(
            study_df=subset_df,
            ra=ra,
            dec=dec,
            crop_size=75,
            StackingMethod="median",
            sigma_clip=True,
            bg_sub_method="Wavelet Decomposition",
            cal_stacked_flux=False, # sensor counts for simplicity
            cal_frames_flux=False,
            output_path=os.path.join(project_root, "fits") + "/",
            output_filename=f"st9107_stack_{n}.fits"
        )
        
        # Load the stacked image
        hdul = fits.open(dest_fits_path)
        data = hdul[0].data
        hdul.close()
        
        # Calculate background noise
        center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
        y, x = np.indices(data.shape)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        bg_pixels = data[dist > 25]
        bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
        noise = np.std(bg_pixels)
        noises.append(noise)
        print(f"Frames stacked: {n}, Noise level: {noise:.4f}")
        
if __name__ == "__main__":
    main()
