# API Reference

This page provides the automatic API documentation for all five core modules of `doverAstroPipeline`, generated directly from the docstrings in the source code.

---

## 1. Extract, Transform, Load (`astropipeline_etl`)
Responsible for querying astronomical databases (NoirLab, SIMBAD, 2MASS) and resolving corresponding raw/calibration data files.

::: astropipeline.astropipeline_etl

---

## 2. Correction & Calibration (`astropipeline_correct`)
Performs instrument calibrations, flat-field division, dark frame subtraction, bad-pixel healing, coordinate warping/rectification, and photometric flux calibration.

::: astropipeline.astropipeline_correct

---

## 3. Metrics & Source Detection (`astropipeline_measure`)
Contains functions for detecting point sources using Wavelet/Difference-of-Gaussians methods, calculating centroids, checking for hot/cold pixels, and managing measurement settings.

::: astropipeline.astropipeline_measure

---

## 4. Stacking & Co-addition (`astropipeline_stack`)
Combines multiple calibrated and aligned frames into a single, high-quality co-added master image, including sigma clipping and background subtraction.

::: astropipeline.astropipeline_stack

---

## 5. Pipeline Orchestration (`astropipeline_manager`)
Orchestrates the individual processing steps (correction, rectification, and calibration) into a unified pipeline flow, supporting concurrent execution.

::: astropipeline.astropipeline_manager
