import numpy as np
import pytest

from stats_arrays.distributions import GeneralizedExtremeValueUncertainty as GEVU
from stats_arrays.errors import InvalidParamsError


def _make_params_array(length=1):
    params = np.zeros(
        (length,),
        dtype=[
            ("input", "u4"),
            ("output", "u4"),
            ("loc", "f4"),
            ("negative", "b1"),
            ("scale", "f4"),
            ("shape", "f4"),
            ("minimum", "f4"),
            ("maximum", "f4"),
        ],
    )
    params["minimum"] = params["maximum"] = np.nan
    params["loc"] = params["scale"] = 1
    return params


@pytest.fixture()
def make_params_array():
    return _make_params_array


def test_random_variables(make_params_array):
    params = make_params_array()
    params["loc"] = 2
    params["scale"] = 5
    # Formula for median (loc - scale * ln ln 2)
    expected_median = 2 - 5 * np.log(np.log(2))
    results = GEVU.random_variables(params, 10000)
    found_median = np.median(results)
    assert results.shape == (1, 10000)
    assert 0.9 * expected_median < found_median
    assert found_median < 1.1 * expected_median

def test_loc_validation(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    with pytest.raises(InvalidParamsError):
        GEVU.validate(params)

def test_scale_validation(make_params_array):
    params = make_params_array()
    params["scale"] = -1
    with pytest.raises(InvalidParamsError):
        GEVU.validate(params)

def test_shape_validation(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    with pytest.raises(InvalidParamsError):
        GEVU.validate(params)
