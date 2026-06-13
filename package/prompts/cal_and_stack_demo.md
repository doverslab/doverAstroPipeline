**Goal**
Create a new notebook in C:\Users\flann\OneDrive\Documents\gits\doverAstroPipeline\package\dev_notebooks called fits_cal_and_stack.ipynb that demonstrates all of the added functionality. 

**Pre-hooks**
Make sure that this process is actually pulling raw FITS files and not just using already processed files within the local_fits folder or within the online archives.


**Requirements**
Use astropipeline_etl.py to retrieve a set of raw images from a single night that all correspond to a single target and filter.
Use astropipeline_correct.py to perform rectilinear correction to each single frame before stacking.
Use astropipeline_stack.py to stack the corrected images using a variety of settings. Demonstrate the different background subtraction methods, sigma clipping, and calibration functions. 
Select 10 celestial objects from the stacked image that span the range of flux values of catalog objects within the frame. Generate a data summary showing how the different correction and calibration techniques affected the SNR of those 10 objects.
Required method comparisons include:
    - Pre vs post-stack calibration
    - Wavelet vs polynomial background subtraction methods
Generate one plot for each method comparison. This plot should be a bar chart.
    - There should be a bar for each celestial object in each method with which it was processed. Example: if sol was used, there would be a bar for sol_pre_cal and sol_post_cal.
    - The bars should be grouped by celestial object.
    - There should be a line plotted across the bars indicating the average SNR for that celestial object.
    - The x-axis should be labeled with the celestial object names.
    - The y-axis should be labeled with the SNR values.

**Artifacts**
- A Jupyter Notebook containing the demonstration code
- A FITS file containing the best stacked image (with a descriptive header)
    - "Best" is defined as the stacked image with the highest average SNR.
- A PNG image of each plot.
- A PNG image for each of the top 3 SNR targets from the best stacked image. Each image should be dynamically scaled for the target and should include a visual marker of the celestial object's position. Zoomed in to 3x3 arcminutes. Provide the name of the target in the filename.

**Post-hooks**
After this process has executed, create another notebook called fits_measure_2.ipynb that mimics the process used in fits_measure.ipynb but using the best stacked image from this process.