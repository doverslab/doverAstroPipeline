import pytest
import pandas as pd
from unittest.mock import patch

from package.src.astropipeline import astropipeline_etl as aple

@pytest.fixture(scope="module", autouse=True)
def mock_noirlab_api():
    with patch('package.src.astropipeline.astropipeline_etl.requests.post') as mock_post:
        class MockResponse:
            def __init__(self, json_data):
                self.json_data = json_data
            def json(self):
                return self.json_data

        def side_effect(url, json=None, **kwargs):
            # Inspect search parameters to mock appropriate response
            search_params = {item[0]: item[1:] for item in json.get("search", [])}
            
            data = {
                "OBJECT": "test_object",
                "url": "http://fake_url/image.fits",
                "archive_filename": "image_ori.fits.fz",
                "prod_type": "image",
                "obs_type": "object",
                "proc_type": "raw",
                "CORN1RA": 0.0, "CORN2RA": 1.0, "CORN3RA": 1.0, "CORN4RA": 0.0,
                "CORN1DEC": 0.0, "CORN2DEC": 0.0, "CORN3DEC": 1.0, "CORN4DEC": 1.0,
            }
            
            # Adjust mock data based on the requested search
            if "obs_type" in search_params:
                obs_types = search_params["obs_type"]
                if "flat" in obs_types:
                    data["obs_type"] = "flat"
                elif "dark" in obs_types:
                    data["obs_type"] = "dark"
            
            if "prod_type" in search_params and "dqmask" in search_params["prod_type"]:
                data["prod_type"] = "dqmask"
                data["proc_type"] = "instcal"
            
            if "proc_type" in search_params and "instcal" in search_params["proc_type"] and data.get("prod_type") != "dqmask":
                data["proc_type"] = "instcal"

            if data.get("proc_type") == "raw" and data.get("obs_type") == "object":
                return MockResponse([None, data])
            elif data.get("prod_type") == "dqmask":
                return MockResponse([None, data])
            else:
                return MockResponse([None, data, data])
            
        mock_post.side_effect = side_effect
        yield mock_post

@pytest.fixture(scope="module", autouse=True)
def mock_astropy_fits():
    with patch('package.src.astropipeline.astropipeline_etl.fits.open') as mock_fits:
        mock_hdr = {
            "INSTRUME": "newfirm",
            "RAWFILE": "test.fits",
            "PROPID": "123",
            "DTCALDAT": "2025-01-01",
            "FILTER": "KXs"
        }
        mock_hdu0 = type('obj', (object,), {'header': mock_hdr})
        mock_hdu1 = type('obj', (object,), {'header': {"DQMASK": "mask.fits"}})
        mock_fits.return_value = [mock_hdu0, mock_hdu1]
        yield mock_fits

@pytest.fixture(scope="module", autouse=True)
def mock_simbad_vizier():
    with patch('package.src.astropipeline.astropipeline_etl.Simbad.query_tap') as mock_simbad, \
         patch('package.src.astropipeline.astropipeline_etl.Vizier.query_region') as mock_vizier:
         
        mock_simbad.return_value.to_pandas.return_value = pd.DataFrame([{"ra": 1, "dec": 1}, {"ra": 2, "dec": 2}])
        
        # Vizier returns a dict of tables
        mock_vizier.return_value = {"II/246/out": type('obj', (object,), {'to_pandas': lambda: pd.DataFrame([{"RAJ2000": 1, "DEJ2000": 1, "Ksnr": 10}, {"RAJ2000": 2, "DEJ2000": 2, "Ksnr": 10}])})()}
        
        yield

@pytest.fixture(scope="module")
def empty_pipe_study():
    test_pipe_study = aple.PipeStudy(
        telescope="kp4m", instrument="newfirm", exposure=10, filter="KXs",
        max_returns=1
    )
    return test_pipe_study


def test_pipestudy_class(empty_pipe_study: aple.PipeStudy):
    assert isinstance(empty_pipe_study, aple.PipeStudy)


def test_find_instcals(empty_pipe_study: aple.PipeStudy):
    filled_pipe_study = empty_pipe_study.find_instcals()
    assert isinstance(filled_pipe_study, pd.DataFrame)
    assert len(filled_pipe_study) >= 1


@pytest.fixture(scope="module")
def populated_study(empty_pipe_study: aple.PipeStudy):
    filled_study = empty_pipe_study
    filled_study.instcal_fits_df = filled_study.find_instcals()
    return filled_study


def test_pipeline_build(populated_study: aple.PipeStudy):
    pipe_instance = next(iter(populated_study))
    pipeline_df = aple.get_pipeline_df(pipe_instance)
    assert isinstance(pipeline_df, pd.DataFrame)
    assert len(pipeline_df) > 4


@pytest.fixture(scope="module")
def pipe_instance(populated_study: aple.PipeStudy):
    return populated_study.instcal_fits_df.iloc[0]


@pytest.fixture(scope="module")
def populated_pipeline(pipe_instance: pd.DataFrame):
    return aple.get_pipeline_df(pipe_instance)


def test_pipeline_includes_raw(populated_pipeline: pd.DataFrame):
    print(sum(populated_pipeline["prod_type"] == "image"))
    assert (
        sum(
            (populated_pipeline["prod_type"] == "image")
            & (populated_pipeline["obs_type"] == "object")
            & (populated_pipeline["proc_type"] == "raw")
        )
        == 1
    )


def test_pipeline_includes_flat(populated_pipeline: pd.DataFrame):
    assert (
        sum(
            (populated_pipeline["prod_type"] == "image")
            & (populated_pipeline["obs_type"] == "flat")
            & (populated_pipeline["proc_type"] == "raw")
        )
        >= 1
    )


def test_pipeline_includes_dark(populated_pipeline: pd.DataFrame):
    assert (
        sum(
            (populated_pipeline["prod_type"] == "image")
            & (populated_pipeline["obs_type"] == "dark")
            & (populated_pipeline["proc_type"] == "raw")
        )
        >= 1
    )


def test_pipeline_includes_dqm(populated_pipeline: pd.DataFrame):
    assert (
        sum(
            (populated_pipeline["prod_type"] == "dqmask")
            & (populated_pipeline["proc_type"] == "instcal")
        )
        == 1
    )


def test_get_stars_simbad(pipe_instance: pd.DataFrame):
    stars_df = aple.get_catalog_stars(pipe_instance, "SIMBAD")
    assert isinstance(stars_df, pd.DataFrame)
    assert len(stars_df) >= 1

def test_get_stars_2mass(pipe_instance: pd.DataFrame):
    # Mocking since it falls back to a different process
    stars_df = aple.get_catalog_stars(pipe_instance, "2MASS")
    assert isinstance(stars_df, pd.DataFrame)

def test_pipe_file_paths():
    row = pd.DataFrame([{"archive_filename": "test_ori.fits.fz", "url": "https://astroarchive.noirlab.edu/api/retrieve/test_ori.fits.fz/"}])
    paths = aple.PipeFilePaths(row, "./fits/", "study")
    assert paths.pipe_file_path == "./fits/test_study_pipe.csv"
    assert paths.local_fits_path == "./fits/test_study.fits.fz"
    assert paths.raw_url == "https://astroarchive.noirlab.edu/api/retrieve/test_ori.fits.fz/"

def test_find_instcals_empty(empty_pipe_study):
    # Mock requests.post to return empty results
    with patch("package.src.astropipeline.astropipeline_etl.requests.post") as mock_post:
        mock_post.return_value.json.return_value = [None]
        df = empty_pipe_study.find_instcals()
        assert len(df) == 0

def test_get_pipeline_missing(populated_pipeline: pd.DataFrame):
    # Test get_pipeline_df when it returns -1 due to missing files
    row = populated_pipeline.iloc[0]
    
    # Actually get_pipeline_df queries the API via find_precal_match
    # If API returns empty, it will return -1.
    with patch("package.src.astropipeline.astropipeline_etl.requests.post") as mock_post:
        # Return empty list
        mock_post.return_value.json.return_value = []
        res = aple.get_pipeline_df(row)
        assert res == -1

def test_get_pipeline_file(tmp_path):
    df = pd.DataFrame([{"a": 1}])
    p = tmp_path / "test_pipe.csv"
    df.to_csv(p, index=False)
    loaded = aple.get_pipeline_file(str(p))
    assert loaded.iloc[0]["a"] == 1

def test_get_study_file(tmp_path):
    df = pd.DataFrame([{"b": 2}])
    p = tmp_path / "test_study.csv"
    df.to_csv(p, index=False)
    loaded = aple.get_study_file(str(p))
    assert loaded.iloc[0]["b"] == 2

def test_find_precal_match_error(populated_pipeline):
    row = populated_pipeline.iloc[0]
    with patch("package.src.astropipeline.astropipeline_etl.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"errorMessage": "Some error"}
        res = aple.find_precal_match(row, "raw")
        assert res == -1

def test_find_precal_match_ambiguous(populated_pipeline):
    row = populated_pipeline.iloc[0]
    with patch("package.src.astropipeline.astropipeline_etl.requests.post") as mock_post:
        mock_post.return_value.json.return_value = [None, {"url": "url1"}, {"url": "url2"}]
        res = aple.find_precal_match(row, "raw")
        assert res == -1


