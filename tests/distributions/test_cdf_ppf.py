"""``cdf`` and ``ppf`` of the distributions that gained them in #63."""

import numpy as np
import pytest
from scipy import stats

from stats_arrays.distributions import (
    GammaUncertainty,
    GeneralizedExtremeValueUncertainty,
    StudentsTUncertainty,
    UncertaintyBase,
    WeibullUncertainty,
)

PERCENTAGES = np.array([0.01, 0.2, 0.5, 0.8, 0.99])

# (distribution, params rows, frozen SciPy distribution for each row)
CASES = {
    "weibull": (
        WeibullUncertainty,
        [{"loc": 1, "scale": 2, "shape": 1.5}, {"scale": 0.5, "shape": 3}],
        [
            stats.weibull_min(1.5, loc=1, scale=2),
            stats.weibull_min(3, loc=0, scale=0.5),
        ],
    ),
    "gamma": (
        GammaUncertainty,
        [{"loc": 0.5, "scale": 1, "shape": 2}, {"scale": 3, "shape": 0.7}],
        [stats.gamma(2, loc=0.5, scale=1), stats.gamma(0.7, loc=0, scale=3)],
    ),
    "gev": (
        GeneralizedExtremeValueUncertainty,
        [{"loc": 0, "scale": 1, "shape": 0}, {"loc": -3, "scale": 4, "shape": 0}],
        [stats.gumbel_r(loc=0, scale=1), stats.gumbel_r(loc=-3, scale=4)],
    ),
    "student": (
        StudentsTUncertainty,
        [{"shape": 3}, {"loc": 2, "scale": 0.5, "shape": 10}],
        [stats.t(3, loc=0, scale=1), stats.t(10, loc=2, scale=0.5)],
    ),
}


def make_params(rows, **extra):
    return UncertaintyBase.from_dicts(*({**row, **extra} for row in rows))


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_ppf_matches_scipy(case):
    distribution, rows, frozen = case
    params = make_params(rows)
    distribution.validate(params)
    percentages = np.tile(PERCENTAGES, (len(rows), 1))
    result = distribution.ppf(params, percentages)
    assert result.shape == percentages.shape
    for row, dist in enumerate(frozen):
        assert np.allclose(result[row], dist.ppf(PERCENTAGES))


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_cdf_inverts_ppf(case):
    distribution, rows, frozen = case
    params = make_params(rows)
    values = np.vstack([dist.ppf(PERCENTAGES) for dist in frozen])
    original = values.copy()
    result = distribution.cdf(params, values)
    assert np.allclose(result, np.tile(PERCENTAGES, (len(rows), 1)))
    assert np.array_equal(values, original), "cdf changed its input"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_one_dimensional_inputs(case):
    distribution, rows, _ = case
    params = make_params(rows)
    result = distribution.ppf(params, np.full(len(rows), 0.3))
    assert result.shape == (len(rows), 1)
    assert np.allclose(distribution.cdf(params, result.ravel()), 0.3)


@pytest.mark.parametrize("key", ["weibull", "gamma"])
def test_negative_rows_are_negated(key):
    """Gamma and Weibull negate rows with ``negative`` set in Monte Carlo, so
    ``ppf`` does too, and ``cdf`` reads ``-x`` for them, as the lognormal does."""
    distribution, rows, frozen = CASES[key]
    params = make_params(rows, negative=True)
    percentages = np.tile(PERCENTAGES, (len(rows), 1))
    result = distribution.ppf(params, percentages)
    for row, dist in enumerate(frozen):
        assert np.allclose(result[row], -dist.ppf(PERCENTAGES))
    assert np.allclose(distribution.cdf(params, result), percentages)


@pytest.mark.parametrize("key", ["gev", "student"])
def test_negative_ignored(key):
    """GEV and Student's t ignore ``negative`` in Monte Carlo, and here too."""
    distribution, rows, _ = CASES[key]
    percentages = np.tile(PERCENTAGES, (len(rows), 1))
    assert np.allclose(
        distribution.ppf(make_params(rows, negative=True), percentages),
        distribution.ppf(make_params(rows), percentages),
    )
