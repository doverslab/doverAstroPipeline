# Welcome to doverAstroPipeline

`doverAstroPipeline` is a robust, modular Python library designed for processing raw astronomical image data, performing instrument calibrations (dark and flat frames correction), aligning images with pixel-precision based on catalog data or WCS header parameters, performing flux calibration against catalog stars, and stacking FITS images.

---

## High-Level Purpose

The pipeline automates the transformation of raw FITS images into science-ready, calibrated, and stacked FITS images. Key capabilities include:

* **ETL & Data Ingest:** Seamlessly queries the NOIRLab Astro Archive to discover raw frames, darks, flats, and bad pixel masks matching specified search parameters (e.g., filter, instrument, exposure time).
* **Instrument Calibration:** Applies dark subtraction and flat-field (gain) correction. It also identifies and heals bad or invalid pixels.
* **Astrometric Alignment (Rectification):** Warrans correct coordinates for FITS files using either native WCS header mapping or star catalog matching (2MASS or SIMBAD).
* **Flux Calibration:** Calibrates image intensities to Jy (Janskys) by matching detected sources with catalog stars (e.g., 2MASS J-band/K-band magnitudes) and calculating the photometric zero-point.
* **Co-addition / Stacking:** Stacks multiple processed FITS images into a single master image using robust techniques like sigma clipping, background subtraction (Wavelet Decomposition or Polynomial fitting), and median/mean combining.

---

## Installation & Setup

### Requirements
Ensure you have Python 3.11+ installed. The project relies on standard astronomical packages:
* `astropy` (FITS handling, WCS transformation, Cutout extraction)
* `astroquery` (Simbad and Vizier API queries)
* `numpy` & `scipy` (numerical logic and image interpolation)
* `pandas` (tabular metadata processing)
* `pywt` (discrete wavelet transforms for image denoising/background estimation)

### Environment Setup
If you are running in the workspace, you can activate the local `.venv` or set up a clean virtual environment:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install numpy pandas astropy astroquery scipy PyWavelets pyyaml matplotlib requests
```

---

## Quickstart

To run a trial pipeline stack with a catalog-aligned rectification, you can execute the pre-configured [run_pipeline_trial.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/run_pipeline_trial.py) script:

```powershell
python run_pipeline_trial.py
```

This trial performs the following workflow:
1. **Load Study Metadata:** Reads from `fits/apl_study_dover.csv`.
2. **Rectification (Alignment):** Aligns the loaded FITS images using the `2MASS` star catalog.
3. **Star Querying:** Selects the highest-SNR calibration stars.
4. **Stacking & Background Subtraction:** Median-stacks the images with sigma clipping and Wavelet Decomposition background subtraction.
5. **Output generation:** Saves the stacked FITS file to `fits/stacked_trial_counts.fits` (or `_flux.fits` if flux calibration succeeded) along with a timing log and a preview PNG plot.
