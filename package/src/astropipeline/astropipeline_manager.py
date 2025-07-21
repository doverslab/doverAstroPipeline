import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import numpy as np
import matplotlib.pyplot as plt

import astropipeline_correct as aplc
import astropipeline_etl as aple
import astropipeline_measure as aplm

output_folder = './fits/'
num_darks = 5
num_flats = 5
study_name = 'dover'
mask_keep_list = (0)
max_fits_results = 1

study_output_path = output_folder+'apl_study_'+study_name+'.csv'


if os.path.exists(study_output_path):
    test_study_df = aple.get_study_file(study_output_path)
else:
    test_pipe_study = aple.pipeStudy(telescope="kp4m",
                                     instrument="newfirm",
                                     exposure=10,
                                     filter="KXs",
                                     max_returns=max_fits_results)
    test_study_df = test_pipe_study.find_instcals()
    test_study_df.to_csv(study_output_path)


def correct_subpipe(study_df):
    for index, item in study_df.iterrows():

        pipeline_df = aple.get_pipeline_df(item)
        pipe_indexes = np.arange(0, len(pipeline_df))

        raw_index = pipe_indexes[(pipeline_df['proc_type'] == 'raw') &
                                 (pipeline_df['obs_type'] == 'object')]
        raw_row = pipeline_df.iloc[raw_index]
        pipe_paths = aple.pipe_file_paths(raw_row, output_folder, study_name)

        if os.path.exists(pipe_paths.local_fits_path):
            continue

        dqm_index = pipe_indexes[pipeline_df['prod_type'] == 'dqmask']
        dqm_row = pipeline_df.iloc[dqm_index]
        dqm_url = dqm_row['url'].iloc[0]

        pixel_mask, crop_ranges = aplc.load_mask(dqm_url, mask_keep_list)

        dark_indexes_all = pipe_indexes[pipeline_df['obs_type'] == 'dark']
        dark_urls = pipeline_df.iloc[dark_indexes_all]['url']
        dark_means, _, dark_indexes_used = aplc.get_dark_vals(dark_urls,
                                                              num_darks,
                                                              crop_ranges)
        print('Dark Cal Complete')

        flat_indexes_all = pipe_indexes[pipeline_df['obs_type'] == 'flat']
        flat_urls = pipeline_df.iloc[flat_indexes_all]['url']
        gain_vals, flat_counter, flat_times, flat_indexes_used = \
            aplc.get_gain_vals(dark_means,
                               flat_urls,
                               num_flats,
                               crop_ranges,
                               pixel_mask)

        print('Flat Cal Complete')

        print('Image Correction Starting')
        balanced_fits = aplc.image_uniformity_correct(pipe_paths.raw_url,
                                                      dark_means,
                                                      gain_vals,
                                                      crop_ranges)

        print('--Normalization Finished, Starting Mask Repair.')
        healed_fits = aplc.heal_pixels(balanced_fits,
                                       method="linear",
                                       element_select=[-1])

        print('Image Correction Finished. Saving Files.')
        healed_fits.verify('fix')
        healed_fits.writeto(pipe_paths.local_fits_path, overwrite=True)

        study_df.loc[index, 'out_path'] = pipe_paths.local_fits_path
        study_df.loc[index, 'pipe_path'] = pipe_paths.pipe_file_path
        study_df.to_csv(study_output_path)

        pipeline_df.iloc[
            np.concatenate(
                [
                    raw_index,
                    dark_indexes_all[dark_indexes_used],
                    flat_indexes_all[flat_indexes_used],
                    dqm_index
                ],
                axis=0)
            ].to_csv(pipe_paths.pipe_file_path)

        print('Output FITS saved to: '+pipe_paths.local_fits_path)
        print('Study details saved to: '+study_output_path)

    return study_df


def undistort_subpipe(study_df):

    for study_row in study_df.iterrows():

        fits_in = fits.open(study_row[1]['out_path'])

        for index, hdu in enumerate(fits_in):

            if isinstance(hdu, fits.hdu.image.PrimaryHDU):
                continue

            w = WCS(hdu.header)

            stars_df = aple.get_catalog_stars(
                study_row[1],
                frame=hdu.header['RADESYS'].lower(),
                catalog='2MASS')

            for catalog_star in stars_df.iterrows():

                c = SkyCoord(catalog_star[1]['ra'],
                             catalog_star[1]['dec'],
                             frame=hdu.header['RADESYS'].lower(),
                             unit="deg")

                if not c.contained_by(w):
                    continue

                cutout = Cutout2D(hdu.data, c, (100, 100), mode='trim', wcs=w)

                plt.subplot(1, 2, 1)
                plt.cla()
                plt.imshow(cutout.data)
                plt.gca().invert_yaxis()

                found_coords, img_passed = aplm.wdec_bandpass_find(
                    image=cutout.data,
                    num_sigmas=3,
                    wavelet='db2',
                    start_level=3,
                    stop_level=3)

                plt.subplot(1, 2, 2)
                plt.cla()
                plt.imshow(img_passed, origin='upper', cmap='bone')
                plt.gca().invert_yaxis()

                for lvl_data in found_coords:

                    pix_scale = lvl_data[1]
                    coords = lvl_data[2]

                    plt.scatter(coords[1],
                                coords[0],
                                s=250*pix_scale,
                                edgecolor='cyan',
                                facecolor='none',
                                linewidths=1)

                plt.show()

    return 0


test_study_df = correct_subpipe(test_study_df)

test_study_df = aple.get_study_file(study_output_path)

test_study_df = undistort_subpipe(test_study_df)
