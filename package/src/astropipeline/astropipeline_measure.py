import copy
import pandas as pd
import numpy as np
import pywt
from scipy import ndimage


def crop_fits(fits_img, center_x_y, window_size):
    """
    Crop FITS data and maintain header

    Args:
        fits_img (ndarray): An array of FITS data from HDU List
        center_x_y (list): column, row of center pixel for crop
        window_size (list): columns, rows around location to extract

    Returns:
        fits_section (ndarray): extracted pixel values from FITS
        x_extents (1-d array): column index range of extraction
        y_extents (1-d array): row index range of extraction

    """
    if np.size(window_size) == 1:
        start_x = int(np.floor(center_x_y[0] - window_size // 2))
        stop_x = int(np.ceil(center_x_y[0] + window_size // 2 + 1))
        start_y = int(np.floor(center_x_y[1] - window_size // 2))
        stop_y = int(np.ceil(center_x_y[1] + window_size // 2 + 1))
    else:
        start_x = int(np.floor(center_x_y[0] - window_size[0] // 2))
        stop_x = int(np.ceil(center_x_y[0] + window_size[0] // 2 + 1))
        start_y = int(np.floor(center_x_y[1] - window_size[1] // 2))
        stop_y = int(np.ceil(center_x_y[1] + window_size[1] // 2 + 1))

    fits_section = fits_img[start_y:stop_y, start_x:stop_x]

    x_extents = np.arange(start_x, stop_x)
    y_extents = np.arange(start_y, stop_y)

    return fits_section, x_extents, y_extents


def wdec_bandpass_find(
    image: np.array,
    num_returns=3,
    wavelet: str = "haar",
    start_level: int = 0,
    stop_level: int = 3,
):
    """
    Apply wavelet decomposition bandpass filter, return n max values

    Args:
        image (ndarray): An array of pixel values
        num_returns (int): number of maximum values to return
        wavelet (str): name of wavelet to use for decomposition
        start_level (int): highest resolution decomp to use
        stop_level (int): lowest resolution decomp to use

    Returns:
        fits_section (ndarray): extracted pixel values from FITS
        found coordinates (list): 4-column list containing the bandpass where
            each max value was found, the pixel scaling ration at the given
            level, and the row & column location of each max value

    """

    wdecCoeffs = pywt.wavedec2(image,
                               wavelet=wavelet,
                               mode="reflect",
                               level=stop_level)
    emptyCoeffs = pywt.wavedec2(
        np.zeros_like(image), wavelet=wavelet, mode="reflect", level=stop_level
    )

    detail_level_map = np.arange(stop_level, start_level - 1, -1)

    # zero out the approximation
    wdecCoeffs[0] = tuple([np.zeros_like(v) for v in wdecCoeffs[0]])

    # initialize the dataframe to store coordinates above the threshold
    found_coords = list()

    # loop over details (1 is lowest res details, -1 is highest res details)
    for lvl in range(1, stop_level - start_level + 2, 1):

        # get V and H and D coefficients at lvl
        lvl_pass = emptyCoeffs.copy()
        detail_sum = np.sum(np.square(wdecCoeffs[lvl]), axis=0)
        one_hot_details = detail_sum >= \
            np.partition(detail_sum.flatten(), -3)[-3]
        lvl_pass[lvl] = tuple([one_hot_details for details in lvl_pass[lvl]])

        # determine the coordinate scaling based on level
        pixel_ratio = 2 ** (stop_level - lvl + 1)

        # create an idwt2 representation of the details
        lvl_recon = pywt.waverec2(lvl_pass, wavelet=wavelet)

        # find the n strongest response locations at given level
        coords = np.unravel_index(
            np.argpartition(lvl_recon.flatten(), -num_returns)[-num_returns:],
            lvl_recon.shape,
        )

        found_coords.append([detail_level_map[lvl - 1], pixel_ratio, coords])

    # loop over intermediate levels that can be deleted
    for lvl in range(-1, -start_level, -1):
        wdecCoeffs[lvl] = tuple([np.zeros_like(v) for v in wdecCoeffs[lvl]])

    bandpass_img = pywt.waverec2(wdecCoeffs, wavelet)

    return found_coords, bandpass_img


def swt_bandpass(
    image: np.array,
    wavelet: str = "haar",
    hi_level: int = 0,
    lo_level: int = 3
):
    """
    Apply stationary wavelet bandpass filter

    Args:
        image (ndarray): An array of pixel values
        wavelet (str): name of wavelet to use for decomposition
        hi_level (int): highest resolution decomp to use
        lo_level (int): lowest resolution decomp to use

    Returns:
        bandpass_img (ndarray): reconstruction of the image
    """

    if not pywt.Wavelet(wavelet).orthogonal:
        print("Wavelet provided is not orthogonal. Defaulting to db2.")
        wavelet = ["db2"]

    orig_size = np.shape(image)

    orig_row_slice = slice(0, orig_size[0])
    orig_col_slice = slice(0, orig_size[1])

    buffered_size = np.int32(2 ** (np.floor(np.log2(orig_size)) + 1))

    buffer_addn = buffered_size - np.shape(image)

    buffered_image = np.pad(
        image, pad_width=((0, buffer_addn[0]), (0, buffer_addn[1])),
        mode="reflect"
    )

    swtCoeffs = pywt.swt2(
        buffered_image,
        wavelet=wavelet,
        level=lo_level,
        start_level=hi_level,
        trim_approx=True,
        norm=True,
    )

    swtCoeffs[0] = swtCoeffs[0] * 0
    bandpass_img = pywt.iswt2(swtCoeffs, wavelet=wavelet, norm=True)[
        orig_row_slice, orig_col_slice
    ]
    return bandpass_img


def swt_enhance(
    image: np.array,
    wavelet: list = "exhaustive",
    max_level: int = 3,
    approx_level=-1
):
    """
    Apply high-pass stationary wavelet bandpass filter

    Args:
        image (ndarray): An array of pixel values
        wavelet (str): name(s) of wavelet to use for decomposition;
            if "exhaustive", cycles through all discrete wavelets available
        max_level (int): the lowest resolution SWT to use
        approx_level (int): lowest resolution decomp to use

    Returns:
        bandpass_img (ndarray): reconstruction of the image
    """
    # determine the list of wavelets to cycle through
    if wavelet == "exhaustive":
        wavelet = pywt.wavelist(kind="discrete")
    elif not isinstance(wavelet, list):
        wavelet = list([wavelet])

    # only use orthogonal wavelets
    wavelet = [wave for wave in wavelet if pywt.Wavelet(wave).orthogonal]

    # default to db2 if a non-orthogonal wavelet is provided
    if len(wavelet) == 0:
        print("No orthogonal wavelets provided. Defaulting to db2")
        wavelet = ["db2"]

    # store the original size information for removing buffer later
    orig_size = np.shape(image)
    orig_row_slice = slice(0, orig_size[0])
    orig_col_slice = slice(0, orig_size[1])

    # Buffer the array to avoid boundary issues with wavelet transform
    buffered_size = np.int32(2 ** (np.floor(np.log2(orig_size)) + 1))

    buffer_addn = buffered_size - np.shape(image)

    buffered_image = np.pad(
        image,
        pad_width=((0, buffer_addn[0]), (0, buffer_addn[1])),
        mode="reflect"
    )

    # Setup the list of levels to use for the decomposition and reconstruction
    if approx_level == -1:
        level_list = pd.Index(np.arange(0, max_level + 1), name="Levels")
    else:
        level_list = pd.Index([approx_level], name="Levels")

    # Initialize the dataframe for storing results information
    results = pd.DataFrame(columns=wavelet, index=level_list)

    # loop over wavelets
    for wave in wavelet:

        # loop over the level of SWT
        for lvl in level_list:

            scA_lvl = pywt.swt2(
                buffered_image,
                wavelet=wave,
                level=max_level,
                start_level=lvl,
                trim_approx=True,
                norm=True,
            )

            # Core the values below 3 standard deviations of the mean
            threshold_value = np.mean(
                scA_lvl[0][orig_row_slice, orig_col_slice]
            ) - 3 * np.std(scA_lvl[0][orig_row_slice, orig_col_slice])

            swt_low_pass = pywt.threshold(
                scA_lvl[0], value=threshold_value, mode="hard", substitute=0
            )

            # reconstruct the approximation for the given level
            swt_approx = pywt.iswt2(
                (swt_low_pass, (scA_lvl[0] * 0, scA_lvl[0] * 0, scA_lvl[0] * 0)
                 ),
                wavelet=wave,
                norm=True,
            )[orig_row_slice, orig_col_slice]

            # subtract the approximation (low pass) to get the high pass
            swt_output = buffered_image[orig_row_slice, orig_col_slice] - \
                swt_approx

            # find the maximum contrast's Z-score
            effective_contrast = (np.max(swt_output) - np.mean(swt_output)) /\
                np.std(swt_output)

            # store the information in the results dataframe
            results.loc[lvl, wave] = effective_contrast

    return swt_output, results


def wvlt_coarse_find(
    image: np.array, num_sigmas=3, wavelet="coif1", lo_level=3, hi_level=1
):
    """
    Use SWT to process image then return x,y of max values

    Args:
        image (ndarray): An array of pixel values
        num_sigmas (int): number of standard deviations above which pixel
                            locations will be returned
        wavelet (str): name(s) of wavelet to use for decomposition
        lo_level (int): the lowest resolution SWT to use
        hi_level (int): highest resolution decomp to use

    Returns:
        found_coords (array): locations (rows,cols) of the values above the
                                threshold
        swt_passed (ndarray): processed version of the image
    """
    # get the swt representation given the inputs
    swt_passed = swt_bandpass(
        image, wavelet=wavelet, hi_level=hi_level, lo_level=lo_level
    )

    # return the pixel locations of all pixels above the statistical threshold
    found_coords = np.argwhere(
        swt_passed > (np.mean(swt_passed) + num_sigmas * np.std(swt_passed))
    )

    return found_coords, swt_passed


def norm_array(raw_array):
    """
    Rescale the values of an array to be between 0 and 1

    Args:
        raw_array (ndarray): An array of pixel values

    Returns:
        normed_array (ndarray): amplitude-scaled version of the array with the
                                    same size as the input array
    """
    normed_array = (raw_array - np.min(raw_array)) / (
        np.max(raw_array) - np.min(raw_array)
    )

    return normed_array


def background_sample_1d(raw_array, exp_locn, buffer_len):
    """
    Sample background of a 1-D array for mean, standard deviation

    Args:
        raw_array (ndarray): A numpy nx1 float array
        exp_locn (int): index of expected location
        buffer_len (list): elements around expected location to ignore

    Returns:
        background_stats (list): (mean, stdev)

    """
    sample_locations = np.array(range(0, len(raw_array), 1))

    background_vals = raw_array[np.abs(sample_locations - exp_locn) >
                                buffer_len]

    background_stats = (np.mean(background_vals), np.std(background_vals))

    return background_stats


def dog_1d(raw_array, sigma_hi=1, sigma_lo=2, mode="reflect"):
    """
    Perform a bandpass using a difference-of-gaussians

    Args:
        raw_array (ndarray): A numpy nx1 float array
        sigma_hi (float): the highest resolution gaussian blur sigma to use
        sigma_lo (float): the lowest resolution gaussian blur sigma to use

    Returns:
        dog_array (ndarray): the DoG bandpass version of the array

    """

    array_lo_gauss = ndimage.gaussian_filter1d(raw_array,
                                               sigma=sigma_lo,
                                               mode=mode)
    array_hi_gauss = ndimage.gaussian_filter1d(raw_array,
                                               sigma=sigma_hi,
                                               mode=mode)

    dog_array = array_hi_gauss - array_lo_gauss

    return dog_array


def background_sample_2d(raw_array, exp_locn, buffer_len):
    """
    Sample background for mean, standard deviation

    Args:
        raw_array (ndarray): A numpy nxm float array
        exp_locn (list): row, column expected location
        buffer_len (list): rows, columns around expected location to ignore

    Returns:
        background_stats (list): (mean, stdev)

    """

    background_vals = raw_array[
        np.abs(np.arange(0, raw_array.shape[0]) - exp_locn[0]) > buffer_len[0],
        np.abs(np.arange(0, raw_array.shape[1]) - exp_locn[1]) > buffer_len[1],
    ]

    background_stats = (np.mean(background_vals), np.std(background_vals))

    return background_stats


def dog_2d(raw_array, sigma_hi=1, sigma_lo=2, mode='reflect'):
    """
    Perform a bandpass using a difference-of-gaussians

    Args:
        raw_array (ndarray): A numpy nxm float array
        sigma_hi (float): the highest resolution gaussian blur sigma to use
        sigma_lo (float): the lowest resolution gaussian blur sigma to use

    Returns:
        dog_array (ndarray): the DoG bandpass version of the array
    """

    array_lo_gauss = ndimage.gaussian_filter(raw_array,
                                             sigma=sigma_lo,
                                             mode=mode)
    array_hi_gauss = ndimage.gaussian_filter(raw_array,
                                             sigma=sigma_hi,
                                             mode=mode)

    dog_array = array_hi_gauss - array_lo_gauss

    return dog_array


def get_adjacent_pixels(raw_array: np.ndarray,
                        exp_locn: int,
                        extent: int = (1, 1),
                        remove_mid: bool = True):
    """
    Get all pixels within a certain extent of a specific point

    Args:
        raw_array (ndarray): A numpy nxm float array
        exp_locn (list of 2 floats): center location
                                        in pixel coordinates (row, col)
        extent (int or list of 2 ints): number of pixels to grab near target.
            If a single int, the extent will be the same for rows and columns.
            If two values are provided, the first will be used for rows,
                the second for columns.
        remove_mid (bool): if True, converts the center location to nan

    Returns:
        adjacent_pixels (ndarray): an array of pixels
                        of size (2*exent+1,2*extent+1)
    """
    adjacent_pixels = copy.deepcopy(raw_array[
        (exp_locn[0]-extent[0]):(exp_locn[0]+extent[0]+1),
        (exp_locn[1]-extent[1]):(exp_locn[1]+extent[1]+1)])

    if remove_mid:
        adjacent_pixels[extent[0], extent[1]] = np.nan

    return adjacent_pixels


def pixel_check_hot(raw_array, exp_locn, bg_stats):
    """
    Determine if a specified pixel is a "hot" pixel

    Args:
        raw_array (ndarray): A numpy nxm float array
        exp_locn (list of 2 floats): suspected hot pixel's location
                                        in pixel coordinates (row, col)
        bg_stats (tuple): mean and standard deviation of wider values

    Returns:
        pixel_flag (int): 0 if pixel is not hot, 1 if pixel is hot
    """
    pixel_flag = 0

    adjacent_pixels = get_adjacent_pixels(raw_array,
                                          exp_locn=exp_locn,
                                          remove_mid=True)

    pixel_val = raw_array[exp_locn[0], exp_locn[1]]
    adjacent_max = np.nanmax(adjacent_pixels)

    if ((adjacent_max - bg_stats[0]) / (pixel_val - bg_stats[0])) < 0.5:
        pixel_flag = 1

    return pixel_flag


def pixel_check_cold(raw_array, exp_locn, min_threshold=1):
    """
    Determine if a specified pixel is a "cold" pixel

    Args:
        raw_array (ndarray): A numpy nxm float array
        exp_locn (list of 2 floats): suspected hot pixel's location
                                        in pixel coordinates (row, col)
        min_threshold (int): number of standard deviations (below mean)
            that should be considered "cold" for pixel values

    Returns:
        pixel_flag (int): 0 if pixel is not cold, 1 if pixel is cold
    """
    pixel_flag = 0

    adjacent_pixels = get_adjacent_pixels(raw_array,
                                          exp_locn=exp_locn,
                                          remove_mid=True)

    # get the statistics of the adjacent pixels
    adjacent_mean = np.nanmean(adjacent_pixels)
    adjacent_std = np.nanstd(adjacent_pixels)

    # if the pixel of concern is below the stat (stdev) threshold, throw flag
    if raw_array[exp_locn[0], exp_locn[1]] < (
        adjacent_mean - min_threshold * adjacent_std
    ):
        pixel_flag = 1

    return pixel_flag


def repair_hot_pixels(array):
    """
    Replace "hot" pixels with the mean of adjacent pixels

    Args:
        raw_array (ndarray): A numpy nxm float array

    Returns:
        array (ndarray): A numpy nxm float array with hot pixels removed
    """
    pixel_flag = 1

    while pixel_flag == 1:

        max_row, max_col = np.unravel_index(np.argmax(array), array.shape)

        bg_stats = (np.mean(array), np.std(array))

        # output information about replacement process
        print("max pixel location: {:.2f}, {:.2f}".format(max_row, max_col))
        print("mean pixel value: {:.2f}".format(bg_stats[0]))
        print("std of pixel value: {:.2f}".format(bg_stats[1]))
        print("--current max value: {:.2f}".format(array[max_row, max_col]))
        pixel_zscore = (array[max_row, max_col] - bg_stats[0]) / bg_stats[1]
        print("--current pixel z-score: {:.2f}".format(pixel_zscore))

        pixel_flag = pixel_check_hot(array,
                                     [max_row, max_col],
                                     bg_stats=bg_stats)

        if pixel_flag == 1:
            array[max_row, max_col] = np.nanmean(
                get_adjacent_pixels(array,
                                    exp_locn=(max_row, max_col),
                                    remove_mid=True)
            )

    return array


def repair_cold_pixels(array, min_threshold):
    """
    Replace "cold" pixels with the mean of adjacent pixels

    Args:
        raw_array (ndarray): A numpy nxm float array

    Returns:
        array (ndarray): A numpy nxm float array with cold pixels removed
    """
    pixel_flag = 1

    while pixel_flag == 1:

        min_row, min_col = np.unravel_index(np.argmin(array), array.shape)

        bg_stats_global = (np.mean(array), np.std(array))

        adjacent_pixels = get_adjacent_pixels(array,
                                              (min_row, min_col),
                                              remove_mid=True)

        bg_stats_local = (np.nanmean(adjacent_pixels),
                          np.nanstd(adjacent_pixels))

        # output information about replacement process
        print("min pixel location: {:.2f}, {:.2f}".format(min_row, min_col))
        print("mean pixel value (global): {:.2f}".format(bg_stats_global[0]))
        print("std of pixel value (global): {:.2f}".format(bg_stats_global[1]))
        print("--current min value: {:.2f}".format(array[min_row, min_col]))
        pixel_zscore_global = (
            array[min_row, min_col] - bg_stats_global[0]
        ) / bg_stats_global[1]
        pixel_zscore_local = (
            array[min_row, min_col] - bg_stats_local[0]
        ) / bg_stats_local[1]
        print("--current pixel z-score (global): {:.2f}"
              .format(pixel_zscore_global))
        print("--current pixel z-score (local): {:.2f}"
              .format(pixel_zscore_local))

        pixel_flag = pixel_check_cold(
            array, [min_row, min_col], min_threshold=min_threshold
        )

        if pixel_flag == 1:

            array[min_row, min_col] = np.nanmean(adjacent_pixels)

    return array


def create_psf(array_size=(21, 21), impulse_amplitude=1, sigma=1,
               noise_amplitde=0):
    """
    Create a pseudo point spread function (PSF)

    Args:
        array_size (tuple): row, column dimensions of array to return
        impulse_amplitude (float): amplitude of impulse (pre-spreading)
        sigma (float): the Gaussian blur shaping parameter (ref Gaussian curve)
        noise_amplitude (float): peak-to-peak amplitude of normally
            distributed noise to apply to array elements. This is applied
            *after* blur of impulse. Mimics pixel read noise.
    Returns:
        psf_array: 2D array of simulated pixel values for the PSF
    """
    response_array = np.zeros(array_size)
    response_array[array_size[0]//2, array_size[1]//2] = impulse_amplitude
    response_array = ndimage.gaussian_filter(response_array, sigma=sigma)
    noise_array = noise_amplitde*np.random.rand(array_size[0], array_size[1])
    psf_array = response_array+noise_array

    return psf_array


def get_centroid(array, exp_loc=-1, extent=-1):
    """
    Find the centroid of a given cluster of pixels (array values)

    Args:
        array (2-d array): mxn array of floats representing pixel vals
        exp_loc (tuple): suspected location of centroid. If not provided,
                            defaults to median element of array.
        extent (tuple): the number of rows, and columns to evaluate beyond the
                            expected location

    Returns:
        row_centroid (float): sub-pixel row location of centroid
        col_centroid (float): sub-pixel column location of centroid
    """
    # if no expected location is given, assume the center of the array
    if exp_loc == -1:
        exp_loc = (array.shape[0]//2, array.shape[1]//2)
    else:
        exp_loc = (round(exp_loc[0]), round(exp_loc[1]))

    # if no extent is given, assume the entire array
    if extent == -1:
        extent = (array.shape[0]//2, array.shape[1]//2)
        correction_apply = 0
    else:
        extent = (extent, extent)

        # crop the array
        array = array[(exp_loc[0]-extent[0]):(exp_loc[0]+extent[0]+1),
                      (exp_loc[1]-extent[1]):(exp_loc[1]+extent[1]+1)]

        # flag to later correct the output indexes to original coordinates
        correction_apply = 1

    # built arrays of row, col locations to use for multiplication
    row_locs = np.arange(1, array.shape[0] + 1)
    col_locs = np.arange(1, array.shape[1] + 1)

    # multiply summed columns by row locations to get row centroid
    row_centroid = (
        np.sum(np.multiply(row_locs, np.sum(array, axis=1))
               ) / np.sum(array) - 1
    ) - (extent[0] + exp_loc[0])*correction_apply

    # multiply summed rows by col locations to get col centroid
    col_centroid = (
        np.sum(np.multiply(col_locs, np.sum(array, axis=0))
               ) / np.sum(array) - 1
    ) - (extent[1] + exp_loc[1])*correction_apply

    return row_centroid, col_centroid


def get_pix_distances(array, exp_locn):
    """
    Build an array where each value is the distance to a single coordinate

    Args:
        array (2-d array): mxn array of floats representing pixel vals
        exp_locn (tuple of floats): location to get distances from

    Returns:
        distances (2-d array): mxn array of floats representing L2 distances
        C (2-d array) = meshgrid column locations
        R (2-d array) = meshgrid row locations
    """
    C, R = np.meshgrid(np.arange(0, array.shape[0]),
                       np.arange(0, array.shape[1]))
    distances = np.sqrt(np.square(R-exp_locn[0]) +
                        np.square(C-exp_locn[1]))

    return distances, C, R


def get_border_stats(array):
    """
    Get the mean and standard deviation of pixels at the border of an array

    Args:
        array (2-d array): mxn array of floats representing pixel vals

    Returns:
        border_stats (list): mean and standard deviation of border values
    """
    border_vals = np.concatenate(
        (np.ravel(array[[0, -1],]),
         np.ravel(array[1:-1, [0, -1]]))
        )

    border_mean = np.mean(border_vals)
    border_std = np.std(border_vals)

    border_stats = [border_mean, border_std]

    return border_stats
