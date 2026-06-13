# Trial Rectification and Stacking Pipeline Report

This report documents the trial execution of the astronomical image correction and stacking pipeline using the unprocessed FITS files located in the `fits/` directory.

---

## 1. Executive Summary

A trial run of the rectification and stacking subpipelines was executed using the three raw FITS files:
*   [kp1852990_dover.fits.fz](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/fits/kp1852990_dover.fits.fz)
*   [kp1853387_dover.fits.fz](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/fits/kp1853387_dover.fits.fz)
*   [kp1853391_dover.fits.fz](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/fits/kp1853391_dover.fits.fz)

Using our enhanced **2D consensus pointing offset algorithm** combined with **DoG (Difference of Gaussians) star extraction** (up to 500 catalog stars queried from the **2MASS** catalog), we successfully rectified several extensions with sub-arcsecond accuracy.

The stacked result was generated and saved as:
*   **FITS Stack:** [stacked_trial_counts.fits](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/fits/stacked_trial_counts.fits)
*   **Intensity Plot:** [stacked_trial_counts.png](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/fits/stacked_trial_counts.png)

---

## 2. Rectification Subpipe Metrics

The catalog-based rectification processes each 2D image extension (detector chip) independently. The table below details the performance of the alignment algorithm across the exposures.

| Exposure File | Extension | Detected Offset (dx, dy) [px] | Matches (Inliers) | Mean Error [arcsec] | RMSE [arcsec] | Status / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **kp1852990_dover** | Ext 1 | `-1106.0, -638.0` | 6 | `0.278` | `0.298` | **Success** (Moderate confidence) |
| | Ext 2 | `+2060.0, +866.0` | 4 | `0.000` | `0.000` | **Spurious** (Overfit on 4 noise/stars) |
| | Ext 3 | `+634.0, +1812.0` | 4 | `0.000` | `0.000` | **Spurious** (Overfit on 4 noise/stars) |
| **kp1853387_dover** | Ext 1 | `-764.0, +76.0` | 5 | `0.241` | `0.281` | **Spurious** (Sub-threshold star count) |
| | Ext 2 | `+178.0, -40.0` | 23 | `0.447` | `0.483` | **Success** (High confidence alignment) |
| | Ext 3 | `-2120.0, +2868.0` | 4 | `0.000` | `0.000` | **Spurious** (Overfit on 4 noise/stars) |
| **kp1853391_dover** | Ext 1 | `+176.0, -46.0` | 21 | `0.540` | `0.621` | **Success** (High confidence alignment) |
| | Ext 2 | N/A | 0 | N/A | N/A | **Failed** (No offset consensus found) |
| | Ext 3 | `-1184.0, +2698.0` | 3 | N/A | N/A | **Failed** (Insufficient matches for RANSAC) |

> [!NOTE]
> *   An RMSE of `< 0.5 arcsec` represents extremely high-precision astronomical alignment (pixel scale of NEWFIRM is ~0.4 arcsec/pixel).
> *   Mean and RMSE error values of `0.000` occur when RANSAC matches exactly 4 stars, as a 2D homography has 8 degrees of freedom and fits 4 points with zero residual. These are indicative of sparse fields yielding spurious coincidences.

---

## 3. Stacking & Flux Calibration

The stacked image was compiled using the **Median** stacking method with **3-sigma clipping** and **Wavelet Decomposition** background subtraction.

### Flux Calibration Failure Detail
During the stacking calibration, the pipeline outputted the following:
`WARNING: Insufficient catalog stars matched (0) for reliable flux calibration.`
`WARNING: Stacked image flux calibration requested but failed. Saved as sensor counts.`

### Root Cause Analysis
1.  **Independent Extension Failures:** Because extensions are processed independently, the extensions with sparse fields (e.g., Ext 2 and Ext 3 of `kp1853391`) failed to find correct catalog alignments and fell back to unrectified frames.
2.  **Mismatched Alignment:** The stacked image combines the first valid 2D array found in each file. Since the first file (`kp1852990`) failed to correctly align Ext 1 (using a spurious offset for stacking input, or retaining unrectified boundaries), the combined stack contains pixel misalignments.
3.  **Centroiding Failures:** `calibrate_flux` searches for stars at their nominal sky coordinates. Due to the residual pointing offsets in the unaligned stack, the actual stars are shifted outside the 20-pixel centroid search radius, leading to 0 matched stars.

---

## 4. Key Engineering Insights & Recommendations

### Insight 1: Catalog Projection Buffer
Discarding catalog stars using `contained_by(wcs)` prior to pointing offset correction was a major bottleneck. Incorporating a 2000-pixel margin allowed the pipeline to resolve telescope pointing offsets up to 1100 pixels (~7.5 arcminutes), which rescued the alignment of `kp1852990` and `kp1853391`.

### Insight 2: Global Exposure Pointing Consensus
Since all 4 detectors of the NEWFIRM focal plane are physically fixed to the same camera body, they experience the **exact same telescope pointing offset** in a single exposure.
*   *Current behavior:* Each extension calculates its offset independently. Sparse fields are prone to matching random noise/stars at completely incorrect offsets (e.g., `dx=2060, dy=866` instead of the true `-1106, -638`).
*   *Recommendation:* Modify `undistort_subpipe` in [astropipeline_manager.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_manager.py) to perform a first-pass offset search across all extensions, find the single most dominant global offset peak `(dx, dy)` for the entire file, and then enforce this offset as a prior for all extensions. This will guarantee that sparse extensions align perfectly.
