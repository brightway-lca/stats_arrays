import numpy as np
import pytest


def _make_params_array(length: int = 1):
    assert isinstance(length, int)
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
    params["scale"] = params["shape"] = np.nan
    return params


@pytest.fixture()
def make_params_array():
    return _make_params_array


def seeded_random():
    def func(seed=111111):
        return np.random.RandomState(111111)


@pytest.fixture()
def biased_params_1d():
    oneDparams = _make_params_array(1)
    oneDparams["minimum"] = 1
    oneDparams["loc"] = 3
    oneDparams["maximum"] = 4
    return oneDparams


@pytest.fixture()
def biased_params_2d():
    params = _make_params_array(2)
    params["minimum"] = 1
    params["loc"] = 3
    params["maximum"] = 4
    return params


@pytest.fixture()
def right_triangle_min():
    params = _make_params_array(1)
    params["minimum"] = 1
    params["loc"] = 1
    params["maximum"] = 4
    return params


@pytest.fixture()
def right_triangle_max():
    params = _make_params_array(1)
    params["minimum"] = 1
    params["loc"] = 4
    params["maximum"] = 4
    return params
