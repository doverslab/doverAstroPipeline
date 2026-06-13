# Stacking and Calibration Feature Plan (stack.md)

This plan outlines the design and implementation of the FITS stacking pipeline, background subtraction methods, and flux calibration inside the `doverAstroPipeline` package.

## Proposed Changes

---

### Astronomical Pipeline Correction Component

We will implement the background subtraction and flux calibration methods in [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py).

#### [MODIFY] [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py)
- Implement `fit_poly_2d(data, degree)`: Fits a 2D polynomial surface of a given degree to the non-NaN pixel coordinates of an image.
- Implement `fit_wavelet_background(data, wavelet, level)`: Fits a background by performing discrete wavelet decomposition, zeroing out all detail coefficients, and reconstructing the image from approximation coefficients.
- Implement `subtract_background(data, method)`:
  - Supports `"None"`, `"Linear"` (uses `fit_poly_2d` with degree 1), `"Polynomial 7D"` (uses `fit_poly_2d` with degree 7), and `"Wavelet Decomposition"`.
- Implement `calibrate_flux(hdu, catalog_stars_df, log_func)`:
  - Matches detected star centroids to catalog star locations.
  - Measures instrumental fluxes using aperture summation (with local background subtraction).
  - Computes zero-points ($ZP_i = m_{\text{cat}} - m_{\text{inst}}$) and zero-point variance.
  - Computes calibration uncertainties (zero-point variance, gain, catalog matching variance) and saves them in accordance with FITS standard 4.0.
  - Scales the image pixel array to Janskys.

---

### Astronomical Pipeline Stacking Component

We will create a new stacking module [astropipeline_stack.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_stack.py).

#### [NEW] [astropipeline_stack.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_stack.py)
- Implement `stack_images(image_paths, method, sigma_clip, bg_sub_method, final_cal, cal_stacked_flux, cal_frames_flux, catalog_stars_df, output_path, output_filename)`:
  - Reads input frames, applies optional final calibration, individual frame flux calibration, and background subtraction.
  - Performs optional sigma clipping on the stack.
  - Performs mean or median stacking.
  - Performs optional flux calibration on the final stacked image.
  - Formats output filenames according to calibration status (e.g. adding suffixes `_flux`, `_counts`, etc.).
  - Writes the stacked FITS image, a PNG preview of the stack, and a detailed step execution log file.

---

### Testing Component

 We will create a new test suite [test_apl_stack.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_stack.py).

#### [NEW] [test_apl_stack.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_stack.py)
- Add tests for 2D polynomial background fitting and subtraction.
- Add tests for wavelet background subtraction.
- Add tests for the stacking pipeline checking:
  - Mean/median stacking behavior.
  - Sigma clipping.
  - Flux calibration (zero-point solving and unit scaling) on individual frames or stacked image.
  - Validation log contents and output PNG preview.

## Verification Plan

### Automated Tests
Run pytest with:
```powershell
$env:PYTHONPATH="."; .venv\Scripts\pytest package/tests/test_apl_stack.py
```
