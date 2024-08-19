import pytest
import numpy as np

from stats_arrays.distributions import NormalUncertainty
from stats_arrays.errors import InvalidParamsError


def test_normal_validate(make_params_array):
    params = make_params_array(1)
    params["scale"] = np.nan
    pytest.raises(InvalidParamsError, NormalUncertainty.validate, params)


def test_normal_ppf(biased_params_1d, biased_params_2d):
    biased_params_1d = biased_params_1d
    params = biased_params_2d
    biased_params_1d["scale"] = 0.8
    params["scale"] = 0.8
    assert np.allclose(
        np.array([[1.97475873, 3, 3.673297]]),
        NormalUncertainty.ppf(biased_params_1d, np.array([[0.1, 0.5, 0.8]])),
    )
    assert np.allclose(
        np.array([[1.97475873], [3]]),
        NormalUncertainty.ppf(params, np.array([0.1, 0.5])),
    )


def test_normal_cdf(biased_params_1d, biased_params_2d):
    biased_params_1d["scale"] = 0.8
    biased_params_2d["scale"] = 0.8
    assert np.allclose(
        np.array([[0.00620967, 0.10564978, 0.5, 0.89435022]]),
        NormalUncertainty.cdf(biased_params_1d, np.array([[1, 2, 3, 4]])),
    )
    assert np.allclose(
        np.array([[0.00620967, 0.10564978, 0.5, 0.89435022]]),
        NormalUncertainty.cdf(biased_params_2d, np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))[
            1, :
        ],
    )


def test_normal_seeded_random(biased_params_1d):
    biased_params_1d["scale"] = 0.8
    assert np.allclose(
        np.array([[1.49734064]]),
        NormalUncertainty.random_variables(biased_params_1d, 1, np.random.RandomState(111111)),
    )


def test_normal_random(biased_params_1d, biased_params_2d):
    biased_params_1d["scale"] = 0.8
    biased_params_2d["scale"] = 0.8
    variables = NormalUncertainty.random_variables(biased_params_1d, size=50000)
    assert variables.shape == (1, 50000)
    assert 2.95 < np.average(variables) < 3.05
    assert 2.95 < np.median(variables) < 3.05
    variables = NormalUncertainty.random_variables(biased_params_2d, size=50000)
    assert variables.shape == (2, 50000)
    assert not np.allclose(variables[0, :], variables[1, :])
    assert 2.95 < np.average(variables[1, :]) < 3.05
    assert 2.95 < np.median(variables[1, :]) < 3.05
    assert 2.95 < np.average(variables[0, :]) < 3.05


def test_normal_statistics(biased_params_1d):
    biased_params_1d["scale"] = 0.8
    assert (
        {
            "upper": 4.5999999046325684,
            "lower": 1.3999999761581421,
            "median": 3.0,
            "mode": 3.0,
            "mean": 3.0,
        } == NormalUncertainty.statistics(biased_params_1d),
    )


def test_normal_bounded_pdf(biased_params_1d):
    biased_params_1d["scale"] = 0.8
    points = np.array([[1, 2, 3, 4]])
    xs, ys = NormalUncertainty.pdf(biased_params_1d.copy(), points)
    assert np.allclose(points, xs)
    assert np.allclose(np.array([0.02191038, 0.22831136, 0.49867784, 0.22831136]), ys)
    xs, ys = NormalUncertainty.pdf(biased_params_1d.copy())
    assert xs.shape == (200,)
    assert ys.shape == (200,)
    assert xs.min() == 1
    assert 3.98 < xs.max() <= 4
    assert ys.min() == 0.021910377331033407
    assert np.allclose(ys.max(), 0.498668095951)


def test_normal_unbounded_pdf(biased_params_1d):
    biased_params_1d["scale"] = 0.8
    biased_params_1d["minimum"] = biased_params_1d["maximum"] = np.nan
    xs, ys = NormalUncertainty.pdf(biased_params_1d.copy())
    assert xs.shape == (200,)
    assert ys.shape == (200,)
    assert np.allclose(xs.min(), 1.23999989033)
    assert np.allclose(xs.max(), 4.74241173267)
    assert np.allclose(np.average(xs), 2.9912058115)
    assert np.allclose(ys.min(), 0.0443432302212)
    assert np.allclose(ys.max(), 0.498677843058)
    assert np.allclose(np.average(ys), 0.276188653511)
