import pytest
import numpy as np
from scipy.special import erf

from stats_arrays.distributions import LognormalUncertainty as LU
from stats_arrays.errors import InvalidParamsError


def pdf(x, mu, sigma):
    return (
        1
        / (x * np.sqrt(2 * np.pi * sigma**2))
        * np.e ** (-((np.log(x) - mu) ** 2) / (2 * sigma**2))
    )

def cdf(x, mu, sigma):
    return 0.5 * (1 + erf((np.log(x) - mu) / np.sqrt(2 * sigma**2)))

def test_pdf_positive():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma})
    assert np.allclose(
        pdf(1.2, mu, sigma),
        # [0] are X values, [1] are Y values
        LU.pdf(pa, np.array((1.2,)))[1],
    )

def test_pdf_negative():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma, "negative": True})

    assert np.allclose(
        pdf(1.2, mu, sigma),
        # [0] are X values, [1] are Y values
        LU.pdf(pa, np.array((-1.2,)))[1],
    )

def test_pdf_bounds():
    pa = LU.from_dicts({"loc": 1, "scale": 0.5, "minimum": 1, "maximum": 2})
    xs, ys = LU.pdf(pa)
    assert xs.min() == 1
    assert xs.max() == 2

    pa = LU.from_dicts({"loc": 1, "scale": 0.5, "minimum": 1})
    xs, ys = LU.pdf(pa)
    assert xs.min() == 1
    assert xs.max() > 2

    pa = LU.from_dicts({"loc": 1, "scale": 0.5, "maximum": 2})
    xs, ys = LU.pdf(pa)
    assert xs.min() < 1
    assert xs.max() == 2

def test_pdf_bounds_negative():
    pa = LU.from_dicts(
        {"loc": 1, "scale": 0.5, "minimum": -2, "maximum": -1, "negative": True}
    )
    xs, ys = LU.pdf(pa)
    assert xs.min() == -2
    assert xs.max() == -1

    pa = LU.from_dicts({"loc": 1, "scale": 0.5, "minimum": -2, "negative": True})
    xs, ys = LU.pdf(pa)
    assert xs.min() == -2
    assert xs.max() > -1

    pa = LU.from_dicts({"loc": 1, "scale": 0.5, "maximum": -1, "negative": True})
    xs, ys = LU.pdf(pa)
    assert xs.min() < -2
    assert xs.max() == -1

def test_cdf_positive():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma})
    assert np.allclose(cdf(1.2, mu, sigma), LU.cdf(pa, np.array((1.2,)))[0])

def test_cdf_negative():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma, "negative": True})
    assert np.allclose(cdf(1.2, mu, sigma), LU.cdf(pa, np.array((-1.2,)))[0])

def test_cdf_multirow():
    pa = LU.from_dicts(
        {"loc": 0.4, "scale": 0.1, "negative": True},
        {"loc": 0.6, "scale": 0.2, "negative": False},
    )
    assert np.allclose(
        cdf(
            np.array((1.2, 1.5)), np.array((0.4, 0.6)), np.array((0.1, 0.2))
        ),
        LU.cdf(pa, np.array((-1.2, 1.5))).ravel(),
    )

def test_ppf_positive():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma})
    cdf_result = cdf(1.3, mu, sigma)
    assert np.allclose(1.3, LU.ppf(pa, np.array((cdf_result,)))[0])

def test_ppf_negative():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma, "negative": True})
    cdf_result = cdf(1.3, mu, sigma)
    assert np.allclose(-1.3, LU.ppf(pa, np.array((cdf_result,)))[0])

def test_validation():
    dicts = [
        {"loc": np.nan, "scale": 0.1},
        {"loc": 0.1, "scale": np.nan},
    ]
    for d in dicts:
        with pytest.raises(InvalidParamsError):
            LU.validate(LU.from_dicts(d))

def test_seeded_random():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma})
    assert np.allclose(
        LU.random_variables(pa, 100, np.random.RandomState(111111)),
        LU.random_variables(pa, 100, np.random.RandomState(111111)),
    )

def test_rng():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    median = np.exp(mu)
    ci_upper_975 = np.exp(mu) * (np.exp(sigma) ** 1.96)
    pa = LU.from_dicts({"loc": mu, "scale": sigma})
    sample = LU.random_variables(pa, size=int(1e5)).ravel()
    sample.sort()
    assert np.allclose(np.median(sample), median, 0.01)
    assert np.allclose(sample[97500], ci_upper_975, 0.01)

def test_rng_negative():
    mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
    pa = LU.from_dicts({"loc": mu, "scale": sigma, "negative": True})
    sample = LU.random_variables(pa, size=100).ravel()
    assert (sample < 0).sum() == 100
