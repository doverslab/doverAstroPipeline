**Goal**
Create a new python file that contains functions to stack FITS images. Call the new file astropipeline_stack.py.

**Intermediate Goal**
In order to retrieve stackable images, the stacking function will call functions from astropipeline_etl and astropipeline_correct. 

**Inputs**
The stacking process shall allow for the user to specify a variety of parameters, including but not limited to:
- Type of images to stack
- Whether to use median or mean stacking
- Whether to use sigma clipping
- Method of background subtraction to use, with options:
    - None
    - Linear
    - Polynomial 7D
    - Wavelet Decomposition
- All correction should be accomplished using functions within astropipeline_correct.py. New functions may be required in order to accomplish this.
 - Method of final calibration to use, with options:
    - None
    - Master bias and non-uniformity correction
 - An option to calibrate the stacked image to flux units instead of sensor counts. This option should default to correcting to Janskys if the calibration is successful. If the calibration fails, the image should be saved as sensor counts and a warning should be printed to the log file as well as the console.
- An option to calibrate each individual frame to flux units instead of sensor counts before stacking. This option should default to correcting to Janskys if the calibration is successful. If the calibration fails, the image should be saved as sensor counts and a warning should be printed to the log file as well as the console.

**Artifacts**
 - A log file with:
    - Details of the original request
    - The result location of the stacked image
    - Paths to the input frames
    - Time of initial request and completion of each step in the process:
        - Calibration
        - Background subtraction
        - Sigma clipping
        - Stacking
 - A stacked image in FITS format
 - A preview image of the stacked image in PNG format (for user verification)

**Outputs**
- List of FITS image paths used
- Output path for the stacked image
- Output filename
- If the stacked image is calibrated to flux units instead of sensor counts, the output filename should reflect this.
    - The calibration data should be included in the FITS header in accordance with FITS standard 4.0
    - The calibration data should include details about the uncertainty of the calibration based on variance in the zero point and the gain of the detector as well as the variance from star catalog matching.
- If the stacked image is not calibrated to flux units instead of sensor counts, the output filename should reflect this and the log file should contain a warning about this.`
 - If each frame is calibrated to flux units instead of sensor counts before stacking, the output filename should reflect this and the log file should contain a warning about this.
    - The calibration data should be included in the FITS header in accordance with FITS standard 4.0
    - The calibration data should include details about the uncertainty of the calibration based on variance in the zero point and the gain of the detector as well as the variance from star catalog matching.
