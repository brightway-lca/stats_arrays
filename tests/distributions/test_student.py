import pytest
import numpy as np

from stats_arrays.distributions import StudentsTUncertainty
from stats_arrays.errors import InvalidParamsError


def pretty_close(a, b):
    if b == 0:
        assert a - 0.1 < b < a + 0.1
    else:
        assert 0.9 * a < b < 1.1 * a


def test_loc_and_scale_nan(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    sample = StudentsTUncertainty.random_variables(params, 5000)
    pretty_close(np.median(sample), 0)


def test_loc_matters(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    params["loc"] = 10
    sample = StudentsTUncertainty.random_variables(params, 1000)
    pretty_close(np.median(sample), 10)


def test_scale_matters(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    sample_1 = StudentsTUncertainty.random_variables(params, 5000)
    params["scale"] = 1000
    sample_2 = StudentsTUncertainty.random_variables(params, 5000)
    assert np.std(sample_1) < np.std(sample_2)


def test_random_variables(make_params_array):
    params = make_params_array()
    params["shape"] = 5
    sample = StudentsTUncertainty.random_variables(params, 20000)
    # nu / (nu - 2) if nu > 2
    expected_variance = 5.0 / 3
    pretty_close(np.var(sample), expected_variance)


def test_scale_validation(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    # nan is OK
    StudentsTUncertainty.validate(params)
    # > 0 is OK
    params["scale"] = 1
    StudentsTUncertainty.validate(params)
    # <= 0 is not
    params["scale"] = 0
    pytest.raises(InvalidParamsError, StudentsTUncertainty.validate, params)
    params["scale"] = -1
    pytest.raises(InvalidParamsError, StudentsTUncertainty.validate, params)


def test_shape_validation(make_params_array):
    params = make_params_array()
    params["shape"] = 1
    # > 0 is OK
    params["shape"] = 1
    StudentsTUncertainty.validate(params)
    # <= 0 is not
    params["shape"] = 0
    pytest.raises(InvalidParamsError, StudentsTUncertainty.validate, params)
    params["shape"] = -1
    pytest.raises(InvalidParamsError, StudentsTUncertainty.validate, params)
