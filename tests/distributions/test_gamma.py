import numpy as np
import pytest

from stats_arrays.distributions import GammaUncertainty
from stats_arrays.errors import InvalidParamsError


def pretty_close(a, b):
    if b == 0:
        assert a - 0.05 < b < a + 0.05
    else:
        assert 0.9 * a < b < 1.1 * a


def test_random_variables(make_params_array):
    params = make_params_array()
    params["shape"] = 2
    params["scale"] = 5
    sample = GammaUncertainty.random_variables(params, 10000)
    # Mean: shape * scale
    pretty_close(2 * 5, np.mean(sample))
    # Mean: shape * scale^2
    pretty_close(2 * 5**2, np.var(sample))


def test_random_variables_2d(make_params_array):
    params = make_params_array(2)
    params["shape"] = (2, 3)
    params["scale"] = (5, 10)
    sample = GammaUncertainty.random_variables(params, 10000)
    pretty_close(2 * 5, np.mean(sample[0, :]))
    pretty_close(3 * 10, np.mean(sample[1, :]))


def test_random_variables_offset(make_params_array):
    params = make_params_array(2)
    params["shape"] = (2, 3)
    params["scale"] = (5, 10)
    params["loc"] = (100, np.nan)
    sample = GammaUncertainty.random_variables(params, 10000)
    pretty_close(2 * 5 + 100, np.mean(sample[0, :]))
    pretty_close(3 * 10, np.mean(sample[1, :]))


def test_loc_nan_ok(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = 1
    params["shape"] = 1
    GammaUncertainty.validate(params)


def test_scale_validation(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = -1
    params["shape"] = 1
    with pytest.raises(InvalidParamsError):
        GammaUncertainty.validate(params)


def test_shape_validation(make_params_array):
    params = make_params_array()
    params["loc"] = np.nan
    params["scale"] = 1
    params["shape"] = -1
    with pytest.raises(InvalidParamsError):
        GammaUncertainty.validate(params)
