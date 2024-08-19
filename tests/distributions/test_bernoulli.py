import numpy as np
import pytest

from stats_arrays.distributions import BernoulliUncertainty


def test_bernoulli_ppf(biased_params_1d, biased_params_2d):
    assert np.allclose(
        BernoulliUncertainty.ppf(biased_params_1d, np.array([[0, 0.25, 0.5, 0.75, 1]])),
        np.array([[1, 1, 1, 4, 4]]),
    )
    assert np.allclose(
        BernoulliUncertainty.ppf(biased_params_2d, np.array([0.5, 0.8])),
        np.array([[1], [4]]),
    )


def test_bernoulli_cdf(biased_params_1d, biased_params_2d):
    assert np.allclose(
        BernoulliUncertainty.cdf(biased_params_1d, np.array([[1, 2, 3, 4]])),
        np.array([[0, 0, 1, 1]]),
    )
    assert np.allclose(
        BernoulliUncertainty.cdf(biased_params_2d, np.array([1, 3])),
        np.array([[0], [1]]),
    )


def test_bernoulli_seeded_random(biased_params_1d):
    assert np.allclose(
        BernoulliUncertainty.random_variables(
            biased_params_1d, 10, np.random.RandomState(111111)
        ),
        BernoulliUncertainty.random_variables(
            biased_params_1d, 10, np.random.RandomState(111111)
        ),
    )


def test_bernoulli_random(biased_params_1d, biased_params_2d):
    variables = BernoulliUncertainty.random_variables(biased_params_1d, 50000)
    assert 0.3 < np.average(variables) < 0.35
    assert variables.shape == (1, 50000)
    variables = BernoulliUncertainty.random_variables(biased_params_2d, 50000)
    assert 0.3 < np.average(variables[0, :]) < 0.35
    assert 0.3 < np.average(variables[1, :]) < 0.35
    assert variables.shape == (2, 50000)
    assert not np.allclose(variables[0, :], variables[1, :])


def test_bernoulli_statistics(biased_params_1d):
    bern_stats = {
        "upper": None,
        "lower": None,
        "median": None,
        "mode": None,
        "mean": 3,
    }
    assert BernoulliUncertainty.statistics(biased_params_1d) == bern_stats


def test_bernoulli_pdf(biased_params_1d):
    with pytest.raises(NotImplementedError):
        BernoulliUncertainty.pdf(biased_params_1d)
