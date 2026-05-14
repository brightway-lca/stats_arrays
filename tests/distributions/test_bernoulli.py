import numpy as np
import pytest

from stats_arrays.distributions import BernoulliUncertainty
from stats_arrays.errors import InvalidParamsError


@pytest.fixture()
def bernoulli_params_1d():
    """Params for Bernoulli test with loc=0.3 (30% chance of 1)."""
    params = np.zeros(
        (1,),
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
    params["loc"] = 0.3
    return params


@pytest.fixture()
def bernoulli_params_2d():
    """Params for Bernoulli test with loc=0.3 for two rows."""
    params = np.zeros(
        (2,),
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
    params["loc"] = [0.3, 0.7]
    return params


def test_bernoulli_ppf(bernoulli_params_1d, bernoulli_params_2d):
    """PPF returns 1 when q > 1-p, else 0."""
    # For loc=0.3: threshold is 1-0.3=0.7; q > 0.7 → 1, q <= 0.7 → 0
    result = BernoulliUncertainty.ppf(
        bernoulli_params_1d, np.array([[0, 0.25, 0.5, 0.75, 1]])
    )
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    assert np.allclose(result, expected)

    # For loc=[0.3, 0.7]: thresholds are 0.7 and 0.3 respectively
    result = BernoulliUncertainty.ppf(
        bernoulli_params_2d, np.array([[0.2, 0.5], [0.2, 0.5]])
    )
    # Row 0 (p=0.3, threshold=0.7): 0.2 <= 0.7 → 0, 0.5 <= 0.7 → 0
    # Row 1 (p=0.7, threshold=0.3): 0.2 <= 0.3 → 0, 0.5 > 0.3 → 1
    expected = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert np.allclose(result, expected)


def test_bernoulli_cdf(bernoulli_params_1d, bernoulli_params_2d):
    """CDF is 0 for x<0, (1-p) for 0<=x<1, 1 for x>=1."""
    # For loc=0.3: CDF is 0 below 0, 0.7 in [0,1), 1 at and above 1
    assert np.allclose(
        BernoulliUncertainty.cdf(
            bernoulli_params_1d, np.array([[-1, 0, 0.5, 1]])
        ),
        np.array([[0.0, 0.7, 0.7, 1.0]]),
    )
    # For loc=[0.3, 0.7]: each row evaluated at x=0.5 (which is in [0,1))
    # Row 0 (p=0.3): CDF(0.2) = 1-0.3 = 0.7
    # Row 1 (p=0.7): CDF(0.5) = 1-0.7 = 0.3
    assert np.allclose(
        BernoulliUncertainty.cdf(bernoulli_params_2d, np.array([0.2, 0.5])),
        np.array([[0.7], [0.3]]),
    )


def test_bernoulli_seeded_random(bernoulli_params_1d):
    """Test that same seed produces same results."""
    np.random.seed(111111)
    result1 = BernoulliUncertainty.random_variables(
        bernoulli_params_1d, 10, np.random.RandomState(111111)
    )
    np.random.seed(111111)
    result2 = BernoulliUncertainty.random_variables(
        bernoulli_params_1d, 10, np.random.RandomState(111111)
    )
    assert np.allclose(result1, result2)


def test_bernoulli_random(bernoulli_params_1d):
    """Test random variable generation.

    Bernoulli returns 0 or 1 based on random sample <= loc.
    With loc=0.3, ~30% of samples should be 1.
    """
    variables = BernoulliUncertainty.random_variables(bernoulli_params_1d, 50000)
    # ~30% should be 1 (0.3 probability)
    assert 0.25 < np.mean(variables) < 0.35
    assert variables.shape == (1, 50000)
    assert set(np.unique(variables)) == {0.0, 1.0}  # Only 0s and 1s


def test_bernoulli_statistics(bernoulli_params_1d):
    """Statistics returns a basic dict with mean=loc."""
    # UncertaintyBase.statistics returns a basic structure
    result = BernoulliUncertainty.statistics(bernoulli_params_1d)
    assert result["mean"] == 0.3  # loc value
    assert result["mode"] is None
    assert result["median"] is None
    assert result["upper"] is None
    assert result["lower"] is None


def test_bernoulli_pdf(bernoulli_params_1d):
    with pytest.raises(NotImplementedError):
        BernoulliUncertainty.pdf(bernoulli_params_1d)


def test_bernoulli_validation(bernoulli_params_1d):
    """Test validation for Bernoulli distribution."""
    # Valid loc=0.3 should pass
    BernoulliUncertainty.validate(bernoulli_params_1d)

    # Test with loc=0 (should pass)
    params = bernoulli_params_1d.copy()
    params["loc"] = 0.0
    BernoulliUncertainty.validate(params)

    # Test with loc=1 (should pass)
    params = bernoulli_params_1d.copy()
    params["loc"] = 1.0
    BernoulliUncertainty.validate(params)

    # Test with loc < 0 (should fail)
    params = bernoulli_params_1d.copy()
    params["loc"] = -0.1
    with pytest.raises(InvalidParamsError):
        BernoulliUncertainty.validate(params)

    # Test with loc > 1 (should fail)
    params = bernoulli_params_1d.copy()
    params["loc"] = 1.1
    with pytest.raises(InvalidParamsError):
        BernoulliUncertainty.validate(params)
