import numpy as np
import pytest

from stats_arrays.distributions import UniformUncertainty


@pytest.fixture()
def unif_params_1d(make_params_array):
    oneDparams = make_params_array(1)
    oneDparams["minimum"] = 1
    oneDparams["loc"] = 2
    oneDparams["maximum"] = 3
    return oneDparams


@pytest.fixture()
def unif_params_2d(make_params_array):
    params = make_params_array(2)
    params["minimum"] = 1
    params["loc"] = 2
    params["maximum"] = 3
    return params


def test_uniform_ppf(unif_params_1d, unif_params_2d):
    oneDparams = unif_params_1d
    params = unif_params_2d
    assert np.allclose(
        np.array([1, 2, 3]),
        UniformUncertainty.ppf(oneDparams, np.array([[0, 0.5, 1]])),
    )
    assert np.allclose(
        np.array([[1], [2]]), UniformUncertainty.ppf(params, np.array([0, 0.5]))
    )


def test_uniform_cdf(unif_params_1d, unif_params_2d):
    oneDparams = unif_params_1d
    params = unif_params_2d
    assert np.allclose(
        np.array([0, 0.5, 1]),
        UniformUncertainty.cdf(
            oneDparams,
            np.array(
                [
                    [
                        1,
                        2,
                        3,
                    ]
                ]
            ),
        ),
    )
    assert np.allclose(
        np.array([[0], [0.5]]), UniformUncertainty.cdf(params, np.array([1, 2]))
    )


def test_uniform_seeded_random(unif_params_1d):
    oneDparams = unif_params_1d
    assert np.allclose(
        2.15281272,
        UniformUncertainty.random_variables(
            oneDparams, 1, np.random.RandomState(111111)
        ),
    )


def test_uniform_random(unif_params_1d, unif_params_2d):
    oneDparams = unif_params_1d
    params = unif_params_2d
    variables = UniformUncertainty.random_variables(oneDparams, size=5000)
    assert variables.shape == (1, 5000)
    assert 1.95 < np.average(variables) < 2.05
    variables = UniformUncertainty.random_variables(params, size=5000)
    assert variables.shape == (2, 5000)
    assert not np.allclose(variables[0, :], variables[1, :])
    assert 1.95 < np.average(variables[0, :]) < 2.05
    assert 1.95 < np.average(variables[1, :]) < 2.05


def test_uniform_statistics(unif_params_1d):
    oneDparams = unif_params_1d
    assert UniformUncertainty.statistics(oneDparams) == {
        "mean": 2,
        "mode": 2,
        "median": 2,
        "lower": 1,
        "upper": 3,
    }


def test_uniform_pdf(unif_params_1d):
    oneDparams = unif_params_1d
    xs, ys = UniformUncertainty.pdf(oneDparams)
    assert np.allclose(np.array([1, 3]), xs)
    assert np.allclose(np.array([0.5, 0.5]), ys)
    points = np.array([1, 2, 3])
    xs, ys = UniformUncertainty.pdf(oneDparams, points)
    assert np.allclose(points, xs)
    assert np.allclose(np.array([0.5, 0.5, 0.5]), ys)
