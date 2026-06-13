# Prompt: Flux Calibration of FITS Images using Star Catalog Data

This prompt file defines instructions for refactoring and expanding the astronomical pipeline correction module located at [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py) to calibrate pixel values from raw detector counts (ADU) to physical flux units (e.g., Janskys).

---

## 1. Goal & Objectives
Add functions to `astropipeline_correct.py` to calibrate FITS image data to physical units of flux (Janskys) based on comparison of detected stars to star catalog data (such as 2MASS or SIMBAD).
- Compute a photometric zero-point ($ZP$) by matching instrumental star fluxes (aperture/PSF photometry) with their catalog magnitudes.
- Apply the calibration factor to scale the pixel values to Jansky (Jy) units.
- Update FITS headers to reflect the new units and store calibration metadata.

---

## 2a. Modifications to Existing Functions within astropipeline_correct.py
None. This is new functionality that should be mutually exclusive from other correction functions.

## 2b. Modifications to astropipeline_manager.py
Create a new subpipe or function `calibrate_flux_subpipe(study_df)` (or expand existing steps) to query the star catalog, execute the calibration, and save the calibrated images.

---

## 3. Constraints, Edge Cases, & Error Handling
- **Insufficient Stars**: If fewer than 3 catalog stars are successfully matched, abort the calibration. Output a warning to the pipeline log file and console, keeping the image in ADU counts.
- **Logarithmic Math / Negative Pixels**: Safely handle zero or negative pixels when calculating instrumental magnitudes ($m_{\text{inst}} = -2.5 \log_{10}(\text{counts})$).
- **Outlier Rejection**: Use a robust estimator (e.g., RANSAC or Sigma Clipping) to calculate the zero-point, filtering out bad matches, variable stars, or saturated pixels.

---

## 4. New Function Details
- **Aperture Photometry**: A function to measure the sum of pixel values within a specified circular aperture centered on the detected stars, subtracting the local background.
- **Zero-Point Solver**:
  - Solve for the photometric zero-point ($ZP$) using the relationship: $m_{\text{cat}} = m_{\text{inst}} + ZP$.
  - Quality metrics to output to the pipeline log:
    - Number of matched calibrators.
    - Zero-point value and its standard error/dispersion.
- **Calibration Scaling**:
  - Calibrate the image pixels to Jansky units.
  - Set the FITS header keyword `BUNIT = 'Jy'` to indicate Jansky flux units.
  - Add calibration metadata keywords: `PHOTZP` (photometric zero-point), `PHOTSYS` (photometric system, e.g., Vega/AB), and `PHOTPLAM` (pivot wavelength).

---

## 5. Testing Requirements
- **Target Test File**: [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py)
- **New Test Cases**:
  - Test that the zero-point solver correctly calculates $ZP$ given mock instrumental and catalog magnitudes.
  - Test that the image pixels are correctly scaled to Janskys and the `BUNIT` header is updated to `'Jy'`.
  - Test the fallback mechanism when there are insufficient calibration stars.

---

## 6. Context & Reference Code
- Standard conversion between Vega magnitude and Jansky units:
  $$F_{\nu} (\text{Jy}) = F_0 \times 10^{-0.4 \times m}$$
  where $F_0$ is the zero-magnitude flux of the filter band.
- FITS Standard 4.0 guidelines on physical units (`BUNIT`):
  https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
