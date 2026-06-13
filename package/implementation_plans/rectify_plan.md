# Rectilinear Correction Plan (rectify.md)

This implementation plan describes the addition of rectilinear correction functionality to the `doverAstroPipeline` project. It covers modifications to [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py), [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py), and new tests in [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py).

## Proposed Changes

---

### Astronomical Pipeline Correction Component

We will implement the rectilinear correction algorithms inside [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py).

#### [MODIFY] [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py)
- Import `scipy.ndimage as ndimage`, `scipy.spatial.ConvexHull`, and `matplotlib.path.Path`.
- Implement a custom Direct Linear Transform (DLT) homography estimator with coordinate normalization.
- Implement a RANSAC loop (`ransac_homography`) to robustly fit a homography between two coordinate sets.
- Implement `rectify_wcs(hdu, log_func)`:
  - Generates a distortion-free tangent projection (`TAN`) WCS centered on the image center.
  - Maps target pixels to world coordinates, and then back to the original distorted pixel locations.
  - Resamples the image data using `scipy.ndimage.map_coordinates` with bilinear interpolation.
  - Returns the rectified HDU with updated WCS headers.
- Implement `rectify_catalog(hdu, catalog_stars_df, log_func)`:
  - Uses `aplm.wdec_bandpass_find` to locate star peaks in `100x100` cutouts around catalog stars.
  - Refines detected peak coordinates using a local centroid calculation.
  - Uses `ransac_homography` to fit a homography mapping rectified coordinates to original distorted pixel coordinates.
  - Computes alignment metrics: number of matched objects, mean error, and RMSE (in degrees).
  - Resamples the image data using the homography.
  - Masks pixels outside the convex hull of the inlier stars by setting them to `NaN`.
  - Returns the rectified HDU with updated WCS headers and metrics.
- Implement `rectify_image(hdu, method, catalog_stars_df, log_func)` as a unified entry point that implements the fallback mechanism:
  - If `method == "wcs"`, tries `rectify_wcs`.
  - If WCS parsing fails or is invalid (e.g. not celestial), logs a warning and falls back to `method = "catalog"`.

---

### Astronomical Pipeline Manager Component

We will modify [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py) to integrate the new correction algorithms into the pipeline.

#### [MODIFY] [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py)
- Update `undistort_subpipe(study_df, method="wcs", catalog="2MASS")`:
  - Define a logging helper `log_info` that writes to both console and a new log file `fits/pipeline.log`.
  - Open the FITS file at `out_path` for each study row.
  - Loop over all image extensions.
  - Query the catalog stars using `aple.get_catalog_stars`.
  - Call `aplc.rectify_image(hdu, method, catalog_stars_df, log_func)`.
  - Compile the corrected HDUs into a new FITS file saved at `<original_name>_rectified.fits`.
  - Update `study_df` with the path to the rectified FITS file in a new column `rectified_path`.
  - Return the updated `study_df`.

---

### Testing Component

We will add comprehensive unit tests in [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py).

#### [MODIFY] [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py)
- Add `test_rectify_wcs` checking that the WCS correction maps pixels correctly and updates the header with the new target WCS.
- Add `test_rectify_catalog` checking that RANSAC homography correction works, reports alignment metrics, and masks pixels outside the convex hull of inliers.
- Add `test_rectify_image_wcs_corruption_fallback` verifying that if WCS parsing raises an error, the unified function automatically falls back to catalog correction and writes to logs.

## Verification Plan

### Automated Tests
Run pytest with the pythonpath set:
```powershell
$env:PYTHONPATH="."; .venv\Scripts\pytest package/tests/test_apl_correct.py
```

### Manual Verification
Review logs in `fits/pipeline.log` and inspect output FITS structures to verify that corrected images exist and metadata is correctly updated.
