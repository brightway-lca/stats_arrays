import numpy as np
import pytest
from scipy import stats as scipy_stats

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


@pytest.mark.parametrize(
    "minimum, maximum",
    [(5, 10), (5, 9), (0, 2), (-3, 4)],
    ids=["odd count", "even count", "two values", "spans zero"],
)
def test_statistics_match_support(make_params_array, minimum, maximum):
    """Mean and median of the integers minimum..maximum-1.

    Support is [low, high) as in `scipy.stats.randint`, and the mean of the
    integers a..b is (a + b) / 2, so here (minimum + maximum - 1) / 2:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = maximum
    stats = DiscreteUniform.statistics(params)
    support = np.arange(minimum, maximum)
    assert stats["mean"] == scipy_stats.randint(minimum, maximum).mean()
    assert stats["mean"] == support.mean()
    # Midpoint of the two middle values for an even count, as numpy defines it:
    # https://numpy.org/doc/stable/reference/generated/numpy.median.html
    # scipy's `randint.median()` is `ppf(0.5)` and picks the lower one instead.
    assert stats["median"] == np.median(support)
    assert stats["mode"] is None
    assert stats["lower"] == minimum
    assert stats["upper"] == maximum
    assert all(isinstance(stats[k], float) for k in ("mean", "median", "lower", "upper"))


def test_statistics_mean_matches_samples(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 5
    params["maximum"] = 10
    stats = DiscreteUniform.statistics(params)
    sample = DiscreteUniform.random_variables(params, 100000, np.random.RandomState(0))
    assert stats["mean"] == 7.0
    assert np.isclose(sample.mean(), stats["mean"], atol=0.02)
    assert np.median(sample) == stats["median"]


def test_statistics_nan_minimum_defaults_to_zero(make_params_array):
    params = make_params_array(length=1)
    params["maximum"] = 4
    stats = DiscreteUniform.statistics(params)
    assert stats["lower"] == 0.0
    assert stats["mean"] == stats["median"] == 1.5
