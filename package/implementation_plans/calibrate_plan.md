# Flux Calibration Feature Plan (calibrate.md)

This plan outlines the final additions to implement the flux calibration prompt in the `doverAstroPipeline` package, including a dedicated manager subpipe, robust outlier rejection for zero-point calculations, and new tests in `test_apl_correct.py`.

## Proposed Changes

---

### Astronomical Pipeline Correction Component

We will refine the flux calibration routine in [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py).

#### [MODIFY] [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py)
- Refine `calibrate_flux(hdu, catalog_stars_df, log_func)`:
  - Add robust **Sigma Clipping** outlier rejection to filter out anomalous zero-point values ($ZP_i$) before computing the final mean zero-point.
  - Set the header keywords `PHOTSYS = 'AB'` and `PHOTPLAM = 2.2` (in addition to `BUNIT`, `PHOTZP`, etc.) in accordance with FITS standard 4.0.

---

### Astronomical Pipeline Manager Component

We will implement the flux calibration subpipe in [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py).

#### [MODIFY] [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py)
- Implement `calibrate_flux_subpipe(study_df, catalog="2MASS")`:
  - Loops over each study row.
  - Queries catalog stars using `aple.get_catalog_stars`.
  - Executes `aplc.calibrate_flux` on each image extension.
  - Saves the calibrated FITS file with suffix `_calibrated.fits`.
  - Updates the study dataframe with the path `calibrated_path`.

---

### Testing Component

We will add tests to [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py) and [test_apl_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_manager.py).

#### [MODIFY] [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py)
- Add `test_calibrate_flux_success` to check that the zero-point calculation scales pixels correctly to Janskys and sets the required `BUNIT` and metadata keywords.
- Add `test_calibrate_flux_insufficient_stars` to verify that calibration fails gracefully when not enough catalog stars match.

#### [MODIFY] [test_apl_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_manager.py)
- Add `test_calibrate_flux_subpipe` to verify the new manager subpipe execution, catalog query calls, and output FITS generation.

## Verification Plan

### Automated Tests
Run pytest with:
```powershell
$env:PYTHONPATH="."; .venv\Scripts\pytest package/tests/test_apl_correct.py package/tests/test_apl_manager.py
```
