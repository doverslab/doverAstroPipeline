# Prompt: Rectify and Expand `astropipeline_correct.py`

This prompt file defines instructions for modifying, refactoring, and expanding the astronomical pipeline correction module located at [astropipeline_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/src/astropipeline/astropipeline_correct.py).

---

## 1. Goal & Objectives
Add functions to astropipeline_correct.py to perform rectilinear corrections to the FITS image based on either WCS information within the header or using celestial objects within the image as compared to those retrieved from star catalogs, such as SIMBAD.
- 

---

## 2a. Modifications to Existing Functions within astropipeline_correct.py
None. This is new functionality that should be mutually exclusive from other functions.

## 2b. Modifications to astropipeline_manager.py
The new functions within astropipeline_correct.py will be called from within astropipeline_manager. The existing function, undistort_subpipe, should be updated to call the new functions. It already contains architecture to retrieve star catalog data in order to perform rectilinear corrections using the objects found in the image. The undistort subpipe should be expanded to provide optional functionality for rectilinear corrections based on WCS information in the header or based on celestial objects found in the image.

## 3. Constraints, Edge Cases, & Error Handling
If the user specifies that the image should be corrected based on the header data, but the header data is corrupted, automatically switch to correcting based on star catalog data. Output this information to the log file created by the pipeline as well as the console.

## 4. New function details
- The quality of the alignment should be outputted to the log file created by the pipeline. Included metrics required are:
  - Number of matched celestial objects (catalog vs image)
  - Mean error distance (angular subtense)
  - Root-mean-square error (angular subtense)
- If the rectification is successful, the image should be outputted to a new FITS file.
- If the rectification is succesful, the new FITS file should only include pixels within a region of successful rectification. The remaining pixels should be filled with masked values (this should be done in accordance with the FITS file standards).

## 5. Testing Requirements
- **Target Test File:** [test_apl_correct.py](file:///c:/Users/flann/OneDrive/Documents/gits/doverAstroPipeline/package/tests/test_apl_correct.py)
- **New Test Cases:**
  - Test that the image is corrected using WCS data when specified and that the header is properly updated.
  - Test that the image is corrected using star catalog data when specified and that the header is properly updated.
  - Test that the image is corrected using star catalog data when WCS data is corrupted and that this is outputted to the log file and the console. This only needs to be a test that the WCS correction is called after the corruption is detected.

## 6. Context & Reference Code
 - When using WCS data and the star catalog to rectify the image, a RANSAC method should be used with a SolvePnP algorithm (similar to OpenCV's method)
 - Reference code:
  - https://github.com/astronomer/photutils/blob/main/photutils/wcs/core.py
 - All FITS files output by these functions should be written in accordance with the 2018 FITS standard, which is available here:
  https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf


## 7. Revision Notes
 - An initial band-pass filtering scheme should be applied to images before they undergo correction based on catalog star alignment. Use extract_star_samples from astropipeline_measure to accomplish this. This needs to be done so that there is a better chance of finding enough stars.
  - When querying the star catalog, to find a set of stars to match with the image, expect that the image will only contain 10% of the stars listed within the catalog. Therefore, you should query an excessive number of stars from the catalog to be confident in finding matches within the image.