import pytest
import pandas as pd

from package.src.astropipeline import astropipeline_etl as aple


@pytest.fixture(scope="module")
def empty_pipe_study():
    test_pipe_study = aple.pipeStudy(
        telescope="kp4m", instrument="newfirm", exposure=10, filter="KXs",
        max_returns=1
    )
    return test_pipe_study


def test_pipestudy_class(empty_pipe_study: aple.pipeStudy):
    assert isinstance(empty_pipe_study, aple.pipeStudy)


def test_find_instcals(empty_pipe_study: aple.pipeStudy):
    filled_pipe_study = empty_pipe_study.find_instcals()
    assert isinstance(filled_pipe_study, pd.DataFrame)
    assert len(filled_pipe_study) >= 1


@pytest.fixture(scope="module")
def populated_study(empty_pipe_study: aple.pipeStudy):
    filled_study = empty_pipe_study
    filled_study.instcal_fits_df = filled_study.find_instcals()
    return filled_study


def test_pipeline_build(populated_study: aple.pipeStudy):
    pipe_instance = next(iter(populated_study))
    pipeline_df = aple.get_pipeline_df(pipe_instance)
    assert isinstance(pipeline_df, pd.DataFrame)
    assert len(pipeline_df) > 4


@pytest.fixture(scope="module")
def pipe_instance(populated_study: aple.pipeStudy):
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


def test_get_stars(pipe_instance: pd.DataFrame):
    stars_df = aple.get_catalog_stars(pipe_instance, "nsc_dr2.object")
    assert isinstance(stars_df, pd.DataFrame)
    assert len(stars_df) > 1
