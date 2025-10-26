import numpy as np
import pytest

from stats_arrays.distributions import BetaPERTUncertainty
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError


@pytest.fixture()
def pert_params_1d(make_params_array):
    """Create 1D BetaPERT parameters with A=1, B=2, C=3, lambda=4"""
    params = make_params_array(1)
    params["minimum"] = 1.0  # A
    params["loc"] = 2.0  # B (mean)
    params["maximum"] = 3.0  # C
    params["scale"] = 4.0  # lambda
    return params


@pytest.fixture()
def pert_params_2d(make_params_array):
    """Create 2D BetaPERT parameters"""
    params = make_params_array(2)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params["scale"] = 4.0
    return params


@pytest.fixture()
def pert_params_no_lambda(make_params_array):
    """Create BetaPERT parameters without lambda (should use default=4)"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    # scale (lambda) remains NaN, should use default
    return params


def test_pert_validation_minimum_nan(make_params_array):
    """Test validation fails when minimum is NaN"""
    params = make_params_array(1)
    params["minimum"] = np.nan
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params["scale"] = 4.0

    with pytest.raises(
        InvalidParamsError, match="Real, positive `A` values are required"
    ):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_loc_nan(make_params_array):
    """Test validation fails when loc (mean) is NaN"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = np.nan
    params["maximum"] = 3.0
    params["scale"] = 4.0

    with pytest.raises(
        InvalidParamsError, match="Real, positive `B` values are required"
    ):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_maximum_nan(make_params_array):
    """Test validation fails when maximum is NaN"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = np.nan
    params["scale"] = 4.0

    with pytest.raises(
        InvalidParamsError, match="Real, positive `C` values are required"
    ):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_improper_bounds_min_greater_than_loc(make_params_array):
    """Test validation fails when minimum > loc"""
    params = make_params_array(1)
    params["minimum"] = 3.0  # A > B
    params["loc"] = 2.0
    params["maximum"] = 4.0
    params["scale"] = 4.0

    with pytest.raises(ImproperBoundsError, match="`A <= B <= C` not respected"):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_improper_bounds_loc_greater_than_max(make_params_array):
    """Test validation fails when loc > maximum"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 4.0  # B > C
    params["maximum"] = 3.0
    params["scale"] = 4.0

    with pytest.raises(ImproperBoundsError, match="`A <= B <= C` not respected"):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_min_equals_max(make_params_array):
    """Test validation fails when minimum equals maximum"""
    params = make_params_array(1)
    params["minimum"] = 2.0
    params["loc"] = 2.0
    params["maximum"] = 2.0  # A == C
    params["scale"] = 4.0

    with pytest.raises(ImproperBoundsError, match="`A` and `C` have the same values"):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_lambda_zero_or_negative(make_params_array):
    """Test validation fails when lambda is zero or negative"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0

    # Test lambda = 0
    params["scale"] = 0.0
    with pytest.raises(
        InvalidParamsError, match="Lambda values must be greater than zero"
    ):
        BetaPERTUncertainty.validate(params)

    # Test lambda < 0
    params["scale"] = -1.0
    with pytest.raises(
        InvalidParamsError, match="Lambda values must be greater than zero"
    ):
        BetaPERTUncertainty.validate(params)


def test_pert_validation_valid_params(pert_params_1d):
    """Test validation passes with valid parameters"""
    # Should not raise any exception
    BetaPERTUncertainty.validate(pert_params_1d)


def test_as_beta_conversion(pert_params_1d):
    """Test conversion from PERT parameters to Beta parameters"""
    beta_params = BetaPERTUncertainty._as_beta(pert_params_1d, default_lambda=4.0)

    # For A=1, B=2, C=3, lambda=4:
    # alpha = 1 + lambda * (B-A)/(C-A) = 1 + 4 * (2-1)/(3-1) = 1 + 4 * 0.5 = 3
    # beta = 1 + lambda * (C-B)/(C-A) = 1 + 4 * (3-2)/(3-1) = 1 + 4 * 0.5 = 3
    expected_alpha = 3.0
    expected_beta = 3.0

    assert np.isclose(beta_params["loc"], expected_alpha)
    assert np.isclose(beta_params["shape"], expected_beta)
    assert np.isnan(beta_params["scale"])  # scale should be NaN for Beta distribution


def test_as_beta_conversion_default_lambda(pert_params_no_lambda):
    """Test conversion uses default lambda when not provided"""
    beta_params = BetaPERTUncertainty._as_beta(
        pert_params_no_lambda,
    )

    # Should use default lambda = 4
    expected_alpha = 3.0
    expected_beta = 3.0

    assert np.isclose(beta_params["loc"], expected_alpha)
    assert np.isclose(beta_params["shape"], expected_beta)


def test_as_beta_conversion_custom_lambda(make_params_array):
    """Test conversion with custom lambda value"""
    params = make_params_array(1)
    params["minimum"] = 0.0
    params["loc"] = 1.0
    params["maximum"] = 2.0
    params["scale"] = 6.0  # lambda = 6

    beta_params = BetaPERTUncertainty._as_beta(params, default_lambda=4.0)

    # For A=0, B=1, C=2, lambda=6:
    # alpha = 1 + 6 * (1-0)/(2-0) = 1 + 6 * 0.5 = 4
    # beta = 1 + 6 * (2-1)/(2-0) = 1 + 6 * 0.5 = 4
    expected_alpha = 4.0
    expected_beta = 4.0

    assert np.isclose(beta_params["loc"], expected_alpha)
    assert np.isclose(beta_params["shape"], expected_beta)


def test_random_variables_broadcasting(pert_params_2d):
    """Test random variable generation with broadcasting"""
    results = BetaPERTUncertainty.random_variables(pert_params_2d, 1000)
    assert results.shape == (2, 1000)
    # Mean should be close to loc (B) = 2.0, but allow some tolerance
    assert 1.8 < np.average(results[0, :]) < 2.2
    assert 1.8 < np.average(results[1, :]) < 2.2
    # Values should be within the expected range [1, 3]
    assert np.all(results >= 1.0)
    assert np.all(results <= 3.0)


def test_random_variables_single_row(pert_params_1d):
    """Test random variable generation with single row"""
    results = BetaPERTUncertainty.random_variables(pert_params_1d, 1000)
    assert results.shape == (1, 1000)
    # Mean should be close to loc (B) = 2.0, but allow some tolerance
    assert 1.8 < np.average(results) < 2.2
    # Values should be within the expected range [1, 3]
    assert np.all(results >= 1.0)
    assert np.all(results <= 3.0)


def test_random_variables_different_lambda_values(make_params_array):
    """Test random variables with different lambda values"""
    params = make_params_array(2)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params[0]["scale"] = 2.0  # Lower lambda - more spread
    params[1]["scale"] = 8.0  # Higher lambda - more concentrated

    results = BetaPERTUncertainty.random_variables(params, 1000)
    assert results.shape == (2, 1000)

    # Both should have mean close to 2.0, but allow some tolerance
    assert 1.8 < np.average(results[0, :]) < 2.2
    assert 1.8 < np.average(results[1, :]) < 2.2

    # Values should be within the expected range [1, 3]
    assert np.all(results >= 1.0)
    assert np.all(results <= 3.0)

    # Higher lambda should have lower variance
    assert np.var(results[1, :]) < np.var(results[0, :])


def test_cdf(make_params_array):
    """Test CDF calculation with asymmetric PERT distribution"""
    # Create asymmetric PERT distribution: A=1, B=1.5, C=3 (left-skewed)
    params = make_params_array(1)
    params["minimum"] = 1.0  # A
    params["loc"] = 1.5  # B (closer to minimum, creating left skew)
    params["maximum"] = 3.0  # C
    params["scale"] = 4.0  # lambda

    # Test points within the range [1, 3]
    test_points = np.array([[1.0, 1.5, 2.0, 2.5, 3.0]])
    cdf_values = BetaPERTUncertainty.cdf(params, test_points)

    assert cdf_values.shape == (1, 5)
    # CDF should be 0 at minimum, 1 at maximum
    assert np.isclose(cdf_values[0, 0], 0.0, atol=1e-6)
    assert np.isclose(cdf_values[0, 4], 1.0, atol=1e-6)

    # For left-skewed distribution, CDF at mode (1.5) should be less than 0.5
    # because more probability mass is concentrated on the left side
    assert cdf_values[0, 1] < 0.5  # CDF at mode (1.5)

    # CDF should be monotonically increasing
    for i in range(4):
        assert cdf_values[0, i] <= cdf_values[0, i + 1]

    # Test that CDF values are reasonable for the asymmetric distribution
    assert 0.0 < cdf_values[0, 1] < 0.4  # CDF at mode should be low due to left skew
    assert 0.6 < cdf_values[0, 3] < 1.0  # CDF at 2.5 should be high


def test_cdf_broadcasting(pert_params_2d):
    """Test CDF with multiple rows (broadcasting)"""
    test_points = np.array([[1.5, 2.0, 2.5], [1.8, 2.2, 2.7]])
    cdf_values = BetaPERTUncertainty.cdf(pert_params_2d, test_points)
    
    assert cdf_values.shape == (2, 3)
    # CDF should be between 0 and 1
    assert np.all(cdf_values >= 0.0)
    assert np.all(cdf_values <= 1.0)
    # CDF should be monotonically increasing for each row
    assert cdf_values[0, 0] <= cdf_values[0, 1] <= cdf_values[0, 2]
    assert cdf_values[1, 0] <= cdf_values[1, 1] <= cdf_values[1, 2]


def test_cdf_with_custom_lambda(make_params_array):
    """Test CDF with custom lambda"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params["scale"] = 6.0  # lambda
    
    test_points = np.array([[1.5, 2.0, 2.5]])
    cdf_values = BetaPERTUncertainty.cdf(params, test_points, default_lambda=6.0)
    
    assert cdf_values.shape == (1, 3)
    assert np.all(cdf_values >= 0.0)
    assert np.all(cdf_values <= 1.0)


def test_ppf(pert_params_1d):
    """Test PPF (percent point function) calculation"""
    # Test percentiles
    percentiles = np.array([[0.0, 0.5, 1.0]])
    ppf_values = BetaPERTUncertainty.ppf(pert_params_1d, percentiles)

    assert ppf_values.shape == (1, 3)
    # PPF should return minimum at 0%, maximum at 100%
    assert np.isclose(ppf_values[0, 0], 1.0, atol=1e-6)
    assert np.isclose(ppf_values[0, 2], 3.0, atol=1e-6)
    # PPF at 50% should be close to the mean (2.0)
    assert 1.9 < ppf_values[0, 1] < 2.1


def test_ppf_broadcasting(pert_params_2d):
    """Test PPF with multiple rows (broadcasting)"""
    percentiles = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.95]])
    ppf_values = BetaPERTUncertainty.ppf(pert_params_2d, percentiles)
    
    assert ppf_values.shape == (2, 3)
    # PPF should be within bounds [1, 3]
    assert np.all(ppf_values >= 1.0)
    assert np.all(ppf_values <= 3.0)
    # PPF should be monotonically increasing for each row
    assert ppf_values[0, 0] <= ppf_values[0, 1] <= ppf_values[0, 2]
    assert ppf_values[1, 0] <= ppf_values[1, 1] <= ppf_values[1, 2]


def test_ppf_with_custom_lambda(make_params_array):
    """Test PPF with custom lambda"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params["scale"] = 6.0  # lambda
    
    percentiles = np.array([[0.0, 0.5, 1.0]])
    ppf_values = BetaPERTUncertainty.ppf(params, percentiles, default_lambda=6.0)
    
    assert ppf_values.shape == (1, 3)
    assert np.all(ppf_values >= 1.0)
    assert np.all(ppf_values <= 3.0)


def test_cdf_ppf_roundtrip(pert_params_1d):
    """Test that CDF and PPF are inverse functions"""
    test_points = np.array([[1.5, 2.0, 2.5]])

    # CDF -> PPF roundtrip
    cdf_values = BetaPERTUncertainty.cdf(pert_params_1d, test_points)
    ppf_values = BetaPERTUncertainty.ppf(pert_params_1d, cdf_values)

    assert np.allclose(test_points, ppf_values, atol=1e-6)


def test_statistics(pert_params_1d):
    """Test statistics calculation"""
    stats = BetaPERTUncertainty.statistics(pert_params_1d)

    # For symmetric PERT distribution, mean should equal loc
    assert np.isclose(stats["mean"], 2.0)
    # Mode should equal loc for symmetric distribution
    assert np.isclose(stats["mode"], 2.0)
    # Median should be close to mean for symmetric distribution (if implemented)
    if isinstance(stats["median"], (int, float)):
        assert 1.9 < stats["median"] < 2.1
    # Lower and upper bounds should match minimum and maximum (if implemented)
    if isinstance(stats["lower"], (int, float)):
        assert np.isclose(stats["lower"], 1.0)
    if isinstance(stats["upper"], (int, float)):
        assert np.isclose(stats["upper"], 3.0)


def test_pdf(pert_params_1d):
    """Test PDF calculation"""
    # Test with specific points
    test_points = np.array([1.0, 2.0, 3.0])
    xs, ys = BetaPERTUncertainty.pdf(pert_params_1d, test_points)

    assert np.allclose(xs, test_points)
    assert len(ys) == 3
    # PDF should be 0 at boundaries for Beta distribution
    assert np.isclose(ys[0], 0.0, atol=1e-6)
    assert np.isclose(ys[2], 0.0, atol=1e-6)
    # PDF should be maximum at the mode (mean for symmetric distribution)
    assert ys[1] > ys[0]
    assert ys[1] > ys[2]


def test_pdf_no_points(pert_params_1d):
    """Test PDF calculation without specific points"""
    xs, ys = BetaPERTUncertainty.pdf(pert_params_1d)

    # Should return arrays of x and y values
    assert len(xs) > 0
    assert len(ys) > 0
    assert len(xs) == len(ys)
    # X values should be within the range [1, 3]
    assert np.all(xs >= 1.0)
    assert np.all(xs <= 3.0)


def test_seeded_random(pert_params_1d):
    """Test seeded random number generation"""
    sr = np.random.RandomState(111111)
    result = BetaPERTUncertainty.random_variables(pert_params_1d, 4, seeded_random=sr)

    # Should return deterministic results with same seed
    expected_shape = (1, 4)
    assert result.shape == expected_shape
    # Values should be within bounds
    assert np.all(result >= 1.0)
    assert np.all(result <= 3.0)


def test_seeded_random_with_custom_lambda(pert_params_1d):
    """Test seeded random with custom lambda"""
    sr = np.random.RandomState(222222)
    # Use custom lambda to ensure it's passed through correctly
    result1 = BetaPERTUncertainty.random_variables(
        pert_params_1d, 4, seeded_random=sr, default_lambda=6.0
    )
    sr2 = np.random.RandomState(222222)
    result2 = BetaPERTUncertainty.random_variables(
        pert_params_1d, 4, seeded_random=sr2, default_lambda=6.0
    )
    # Results should be identical with same seed
    assert np.allclose(result1, result2)


def test_default_lambda_parameter():
    """Test that default lambda parameter works correctly"""
    params = np.zeros(
        1,
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
    params["minimum"] = 1.0
    params["loc"] = 2.0
    params["maximum"] = 3.0
    params["scale"] = np.nan  # scale (lambda) is NaN, should use default

    # Test with custom default lambda
    beta_params = BetaPERTUncertainty._as_beta(params, default_lambda=6.0)
    expected_alpha = 4.0  # 1 + 6 * (2-1)/(3-1) = 4
    expected_beta = 4.0  # 1 + 6 * (3-2)/(3-1) = 4

    assert np.isclose(beta_params["loc"], expected_alpha)
    assert np.isclose(beta_params["shape"], expected_beta)


def test_edge_case_minimum_equals_loc(make_params_array):
    """Test edge case where minimum equals loc (left-skewed)"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 1.0  # A = B
    params["maximum"] = 3.0
    params["scale"] = 4.0

    # Should pass validation
    BetaPERTUncertainty.validate(params)

    # Test statistics
    stats = BetaPERTUncertainty.statistics(params)
    # For left-skewed distribution (A=B), mean should be closer to minimum
    assert np.isclose(stats["mean"], 1.333, atol=0.01)
    # Mode might be undefined for edge cases
    if isinstance(stats["mode"], (int, float)):
        assert np.isclose(stats["mode"], 1.0)


def test_edge_case_loc_equals_maximum(make_params_array):
    """Test edge case where loc equals maximum (right-skewed)"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["loc"] = 3.0  # B = C
    params["maximum"] = 3.0
    params["scale"] = 4.0

    # Should pass validation
    BetaPERTUncertainty.validate(params)

    # Test statistics
    stats = BetaPERTUncertainty.statistics(params)
    # For right-skewed distribution (B=C), mean should be closer to maximum
    assert np.isclose(stats["mean"], 2.667, atol=0.01)
    # Mode might be undefined for edge cases
    if isinstance(stats["mode"], (int, float)):
        assert np.isclose(stats["mode"], 3.0)
