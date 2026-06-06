import pytest
import numpy as np
from astropy.io import fits
from package.src.astropipeline import astropipeline_measure as aplm


@pytest.fixture(scope="module")
def test_fits():
    test_img_array = aplm.create_psf(impulse_amplitude=10,
                                     noise_amplitde=1,
                                     sigma=0.1)
    hdu = fits.PrimaryHDU(test_img_array)
    hdulist = fits.HDUList([hdu])
    return hdulist


def test_fits_creation(test_fits: fits.hdu.hdulist.HDUList):
    assert isinstance(test_fits, fits.hdu.hdulist.HDUList)
    assert isinstance(test_fits[0].data, np.ndarray)


def test_get_centroid(test_fits: np.ndarray):
    row_centroid, col_centroid = aplm.get_centroid(
        test_fits[0].data, exp_loc=(10., 10.)
        )
    assert np.abs(row_centroid-10) < 0.1
    assert np.abs(col_centroid-10) < 0.1


def test_norm_array(test_fits):
    normed_array = aplm.norm_array(test_fits[0].data)
    assert np.abs(normed_array[10, 10])-1 < 1e-3


def test_get_pix_distances(test_fits):
    dist_array, _, _ = aplm.get_pix_distances(test_fits[0].data, (0, 0))
    assert np.abs(dist_array[20, 20]-np.sqrt(800)) < 1e-3


def test_get_border_stats(test_fits):
    border_stats = aplm.get_border_stats(test_fits[0].data)
    assert np.abs(border_stats[0]-0.5) < 0.1
    assert np.abs(border_stats[1]-(1/np.sqrt(12)) < 0.1)


@pytest.fixture(scope="module")
def test_coord_list():
    coords = np.unravel_index(np.array([0, 1, 3, 2], dtype=int), shape=[2, 2])
    return coords


def test_get_unique_points(test_coord_list):
    unq_points = aplm.get_unique_points(test_coord_list,
                                        num_required=3,
                                        extent=1.1)
    unq_pnt_iter = iter(unq_points)
    assert next(unq_pnt_iter) == (0, 0)
    assert next(unq_pnt_iter) == (1, 1)
    assert len(unq_points) == 2


