import sys
import os

# Dynamic path resolution to find workspace root
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from package.src.astropipeline import astropipeline_etl as aple
from package.src.astropipeline import astropipeline_correct as aplc
from package.src.astropipeline import astropipeline_manager as aplmgr
from package.src.astropipeline import astropipeline_stack as apls

# 1. Load study file
study_df = aple.get_study_file('fits/apl_study_dover.csv')

# Ensure output folder is correct
aplmgr.output_folder = './fits/'
aplmgr.study_output_path = './fits/apl_study_dover.csv'

# Update out_path to be relative to workspace root
study_df['out_path'] = study_df['out_path'].apply(lambda p: p.replace('./fits/', 'fits/') if p else p)

print("Starting trial rectification using catalog-based alignment...")
study_df_rectified = aplmgr.undistort_subpipe(study_df.copy(), method="catalog", catalog="2MASS")
rectified_paths = study_df_rectified["rectified_path"].tolist()
print("Rectified paths:", rectified_paths)

# 2. Get catalog stars for stacking calibration
row = study_df_rectified.iloc[0]
stars_df = aple.get_catalog_stars(row, catalog="2MASS")
stars_df = stars_df.dropna(subset=["Kmag"]).sort_values("Kmag")
indices = np.linspace(0, len(stars_df) - 1, 10, dtype=int)
selected_stars = stars_df.iloc[indices].reset_index(drop=True)
print(f"Selected {len(selected_stars)} stars for calibration.")

# 3. Stack images
print("Starting stacking process...")
_, dest_path, _ = apls.stack_images(
    image_paths=rectified_paths,
    method="median",
    sigma_clip=True,
    bg_sub_method="Wavelet Decomposition",
    cal_stacked_flux=True,
    cal_frames_flux=False,
    catalog_stars_df=selected_stars,
    output_path="./fits/",
    output_filename="stacked_trial.fits"
)
print("Stacking completed. Saved to:", dest_path)
