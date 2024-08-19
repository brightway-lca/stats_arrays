import numpy as np

from stats_arrays.distributions import DiscreteUniform


def test_array_shape_1d(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 0
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 100)
    assert sample.shape == (1, 100)


def test_array_shape_2d(make_params_array):
    params = make_params_array(length=10)
    params["minimum"] = 0
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 100)
    assert sample.shape == (10, 100)


def test_random_variables(make_params_array):
    params = make_params_array(length=10)
    params["minimum"] = 5
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 10000)
    assert np.unique(sample).tolist() == [5, 6, 7, 8, 9]
