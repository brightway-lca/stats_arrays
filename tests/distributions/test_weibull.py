import pytest
import numpy as np

from stats_arrays.distributions import WeibullUncertainty
from stats_arrays.errors import InvalidParamsError


def pretty_close(a, b):
    if b == 0:
        assert a - 0.05 < b < a + 0.05
    else:
        assert 0.95 * a < b < 1.05 * a


def test_random_variables(make_params_array):
    params = make_params_array()
    params["scale"] = 2  # lambda
    params["shape"] = 5  # k
    sample = WeibullUncertainty.random_variables(params, 10000)
    # Median: lambda * ln(2)^(1/k)
    pretty_close(2 * np.log(2) ** (1 / 5), np.median(sample))


def test_random_variables_2d(make_params_array):
    params = make_params_array(2)
    params["scale"] = (5, 10)
    params["shape"] = (2, 3)
    sample = WeibullUncertainty.random_variables(params, 10000)
    pretty_close(5 * np.log(2) ** (1 / 2), np.median(sample[0, :]))
    pretty_close(10 * np.log(2) ** (1 / 3), np.median(sample[1, :]))


def test_random_variables_offset(make_params_array):
    params = make_params_array(2)
    params["scale"] = (5, 10)
    params["shape"] = (2, 3)
    params["loc"] = (100, np.nan)
    sample = WeibullUncertainty.random_variables(params, 10000)
    pretty_close(100 + 5 * np.log(2) ** (1 / 2), np.median(sample[0, :]))
    pretty_close(10 * np.log(2) ** (1 / 3), np.median(sample[1, :]))


def test_loc_nan_ok(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = 1
    params["shape"] = 1
    WeibullUncertainty.validate(params)


def test_scale_validation(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = -1
    params["shape"] = 1
    with pytest.raises(InvalidParamsError):
        WeibullUncertainty.validate(params)


def test_shape_validation(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = 1
    params["shape"] = -1
    with pytest.raises(InvalidParamsError):
        WeibullUncertainty.validate(params)
