import re
import numpy as np
import pandas as pd
import random
import astropy.io.fits as pyfits
from astropy.nddata import Cutout2D
from scipy.interpolate import LinearNDInterpolator
from astropy.wcs import WCS


def load_mask(dqm_mask_url, keep_list=(0)):

    dqm_fits = pyfits.open(dqm_mask_url)
    crop_ranges = dict()
    pixel_mask = dict()

    for index, hdu in enumerate(dqm_fits):
        if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
            if isinstance(hdu.data, np.ndarray):

                row_range = range(1, np.size(hdu.data, axis=0) + 1)
                col_range = range(1, np.size(hdu.data, axis=1) + 1)

                mask_slice = np.ix_(row_range, col_range)

                pixel_mask[index] = ~np.isin(hdu.data, keep_list)

                crop_ranges[index] = mask_slice

    return pixel_mask, crop_ranges


def get_dark_vals(dark_fits_urls, number_samples, crop_ranges):

    dark_val_arrays = dict()
    dark_val_counters = dict()
    dark_val_means = dict()

    url_indexes = random.choices(
        np.arange(0, len(dark_fits_urls)),
        k=number_samples
        )

    url_list = dark_fits_urls.iloc[url_indexes]

    for url in url_list:
        dark_fits = pyfits.open(url)

        for index, hdu in enumerate(dark_fits):
            if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
                if isinstance(hdu.data, np.ndarray):
                    if index not in dark_val_arrays:
                        dark_val_arrays[index] = hdu.data[crop_ranges[index]]
                        dark_val_counters[index] = 1
                    else:
                        dark_val_arrays[index] = (
                            dark_val_arrays[index] +
                            hdu.data[crop_ranges[index]]
                        )
                        dark_val_counters[index] += 1

    for index in dark_val_counters:
        dark_val_means[index] = dark_val_arrays[index] /\
                                dark_val_counters[index]

    return dark_val_means, dark_val_counters, url_indexes


def get_gain_vals(
    dark_val_means, flat_fits_urls, number_samples, crop_ranges, pixel_mask
):

    flat_val_arrays = dict()
    flat_val_counters = dict()
    gain_vals = dict()
    flat_val_cumtime = dict()

    url_indexes = random.choices(
        np.arange(0, len(flat_fits_urls)),
        k=number_samples
        )
    url_list = flat_fits_urls.iloc[url_indexes]

    for url in url_list:
        flat_fits = pyfits.open(url)

        for index, hdu in enumerate(flat_fits):
            if isinstance(hdu, pyfits.hdu.compressed.CompImageHDU):
                if isinstance(hdu.data, np.ndarray):
                    vals_dark_remove = (
                        hdu.data[crop_ranges[index]] - dark_val_means[index]
                    )
                    if index not in flat_val_arrays:
                        flat_val_arrays[index] = vals_dark_remove
                        flat_val_counters[index] = 1
                        flat_val_cumtime[index] = hdu.header["EXPTIME"]
                    else:
                        flat_val_arrays[index] = (
                            flat_val_arrays[index] + vals_dark_remove
                        )
                        flat_val_counters[index] += 1
                        flat_val_cumtime[index] += hdu.header["EXPTIME"]

    for index in flat_val_counters:
        gain_vals[index] = flat_val_arrays[index] / flat_val_counters[index]
        gain_vals[index][pixel_mask[index]] = np.nan

    return gain_vals, flat_val_counters, flat_val_cumtime, url_indexes


def parse_wat_table(hdu, wat_df=pd.DataFrame()):

    num_axis = hdu.header["NAXIS"]

    for axis in range(1, num_axis + 1):
        wat_bin = ""
        for hdr_line in hdu.header:
            if ("WAT" + str(axis)) in hdr_line:
                wat_bin = wat_bin + hdu.header[hdr_line]

        this_row = len(wat_df)
        wat_df.loc[this_row, "wtype"] = re.search(r"wtype=(\w+)",
                                                  wat_bin).group(1)
        wat_df.loc[this_row, "axtype"] = re.search(r"axtype=(\w+)",
                                                   wat_bin).group(1)
        wat_df.loc[this_row, "dc_vals"] = re.search(
            r'[lngcor|latcor] = "([^"]*)"', wat_bin
        ).group(1)

    return wat_df


def image_uniformity_correct(
    raw_fits_image_url, dark_val_means, gain_vals, crop_ranges
):

    raw_fits = pyfits.open(raw_fits_image_url)
    temp_fits = raw_fits.copy()
    if "RADECSYS" in temp_fits[0].header:
        temp_fits[0].header["RADESYSa"] = temp_fits[0].header["RADECSYS"]
        temp_fits[0].header.remove("RADECSYS")

    temp_fits.writeto("./fits/temp.fits.fz", overwrite=True)

    with pyfits.open("./fits/temp.fits.fz", update=True) as raw_fits:

        balanced_fits = raw_fits.copy()

        balanced_fits[0].verify("fix+exception")
        balanced_fits[4].verify("fix+exception")

        for index, hdu in enumerate(balanced_fits):

            if isinstance(hdu.data, np.ndarray):

                if index > 1:
                    wat_df = parse_wat_table(hdu, wat_df)
                else:
                    wat_df = parse_wat_table(hdu)

                try:
                    wcs = WCS(hdu.header)
                except:
                    print(
                        "OH NO ---- BIG PROBLEMS WITH WCS FOR EXTENSION: " +
                        str(index)
                    )
                    balanced_fits.pop(index)
                    continue

                # Update the FITS header with the cutout WCS
                balanced_fits[index].data = (
                    hdu.data[crop_ranges[index]] - dark_val_means[index]
                ) / gain_vals[index]

                balanced_fits[index].data[gain_vals == 0] = 0
                min_row = min(crop_ranges[index][0])[0]
                max_row = max(crop_ranges[index][0])[0]
                min_col = min(crop_ranges[index][1][0])
                max_col = max(crop_ranges[index][1][0])

                position = ((max_row - min_row) / 2, (max_col - min_col) / 2)
                size = (max_row - min_row, max_col - min_col)

                cutout = Cutout2D(
                    hdu.data,
                    wcs.pixel_to_world(position[0], position[1]),
                    size,
                    mode="trim",
                    wcs=wcs,
                )

                balanced_fits[index].data = cutout.data
                balanced_fits[index].header = cutout.wcs.to_header()

        fix_exception = balanced_fits[0].verify(option="fix+exception")

        print(fix_exception)

        balanced_fits[0].update_header()

        return balanced_fits


def heal_pixels(fits_image, method="mean", element_select=(-1)):

    healed_fits = fits_image.copy()

    for index, hdu in enumerate(fits_image):
        if not ((index in element_select) | (-1 in element_select)):
            continue

        if isinstance(hdu.data, np.ndarray):
            X, Y = np.meshgrid(
                range(0, np.size(hdu.data, 1)), range(0, np.size(hdu.data, 0))
            )

            dq0_mask = np.isnan(hdu.data)  # invalid elements

            x_valid = X[~dq0_mask]
            y_valid = Y[~dq0_mask]

            valid_elements = hdu.data[~dq0_mask]

            if method == "mean":
                hdu.data[np.isnan(hdu.data)] = np.mean(valid_elements)
            elif method == "linear":
                interp = LinearNDInterpolator(
                    list(zip(x_valid, y_valid)), valid_elements
                )
                healed_fits[index].data = interp(X, Y)
            elif method == "quadratic":
                print("Quadratic interpolation not incorporated yet, "
                      "using linear")
                interp = LinearNDInterpolator(
                    list(zip(x_valid, y_valid)), valid_elements
                )
                healed_fits[index].data = interp(X, Y)

    return healed_fits
