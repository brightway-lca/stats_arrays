import numpy as np
import pytest

from stats_arrays.distributions import BetaUncertainty
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError

ALPHA = 3.3
BETA = 2.2
INPUTS = np.array([0.5, 0.6, 0.8]).reshape((1, -1))
PDF = np.array([1.56479181717, 1.82088038112, 1.536047041126])
CDF = np.array([0.30549, 0.47638, 0.8333])


def _make_params_array(length=2):
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
    params[:]["loc"] = ALPHA
    params[:]["shape"] = BETA
    return params


@pytest.fixture()
def make_params_array():
    return _make_params_array


def test_random_variables_broadcasting(make_params_array):
    params = make_params_array()
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (2, 1000)
    assert 0.55 < np.average(results[0, :]) < 0.65
    assert 0.55 < np.average(results[1, :]) < 0.65


def test_random_variables_single_row(make_params_array):
    params = make_params_array(1)
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (1, 1000)
    assert 0.55 < np.average(results) < 0.65


def test_alpha_validation(make_params_array):
    params = make_params_array()
    params["loc"] = 0
    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)


def test_beta_validation(make_params_array):
    params = make_params_array()
    params["shape"] = 0
    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)


def test_cdf(make_params_array):
    params = make_params_array(1)
    calculated = BetaUncertainty.cdf(params, INPUTS)
    assert np.allclose(CDF, calculated, rtol=1e-4)
    assert calculated.shape == (1, 3)


def test_ppf(make_params_array):
    params = make_params_array(1)
    calculated = BetaUncertainty.ppf(params, CDF.reshape((1, -1)))
    assert np.allclose(INPUTS, calculated, rtol=1e-4)
    assert calculated.shape == (1, 3)


def test_pdf(make_params_array):
    params = make_params_array(1)
    calculated = BetaUncertainty.pdf(params, INPUTS)[1]
    assert np.allclose(PDF, calculated)
    assert calculated.shape == (3,)


def test_seeded_random(make_params_array):
    sr = np.random.RandomState(111111)
    params = make_params_array(1)
    params["shape"] = params["loc"] = 1
    result = BetaUncertainty.random_variables(params, 4, seeded_random=sr)
    expected = np.array([0.59358266, 0.84368537, 0.01394206, 0.87557834])
    assert np.allclose(result, expected)


def test_statistics(make_params_array):
    """Test statistics calculation"""
    params = make_params_array(1)
    stats = BetaUncertainty.statistics(params)

    # For alpha=3.3, beta=2.2, expected mean = 3.3/(3.3+2.2) = 0.6
    expected_mean = ALPHA / (ALPHA + BETA)
    assert np.isclose(stats["mean"], expected_mean)

    # Mode is defined when alpha > 1 and beta > 1
    expected_mode = (ALPHA - 1) / (ALPHA + BETA - 2)
    assert np.isclose(stats["mode"], expected_mode)

    # Check that median, lower, upper are "Not Implemented"
    assert stats["median"] == "Not Implemented"
    assert stats["lower"] == "Not Implemented"
    assert stats["upper"] == "Not Implemented"


def test_statistics_edge_cases(make_params_array):
    """Test statistics for edge cases where mode is undefined"""
    # Test alpha = 1 (mode undefined)
    params = make_params_array(1)
    params["loc"] = 1.0
    params["shape"] = 2.0
    stats = BetaUncertainty.statistics(params)
    assert stats["mode"] == "Undefined"

    # Test beta = 1 (mode undefined)
    params["loc"] = 2.0
    params["shape"] = 1.0
    stats = BetaUncertainty.statistics(params)
    assert stats["mode"] == "Undefined"


def test_statistics_with_scaling(make_params_array):
    """Test statistics calculation with minimum/maximum scaling"""
    params = make_params_array(1)
    params["minimum"] = 2.0
    params["maximum"] = 5.0

    stats = BetaUncertainty.statistics(params)

    # Expected mean with scaling: (alpha/(alpha+beta)) * scale + loc
    # = (3.3/5.5) * 3 + 2 = 0.6 * 3 + 2 = 3.8
    expected_mean = (ALPHA / (ALPHA + BETA)) * 3.0 + 2.0
    assert np.isclose(stats["mean"], expected_mean)

    # Expected mode with scaling: ((alpha-1)/(alpha+beta-2)) * scale + loc
    # = (2.3/3.5) * 3 + 2 = 0.657 * 3 + 2 = 3.971
    expected_mode = ((ALPHA - 1) / (ALPHA + BETA - 2)) * 3.0 + 2.0
    assert np.isclose(stats["mode"], expected_mode)


def test_pdf_no_xs(make_params_array):
    """Test PDF calculation without providing xs parameter"""
    params = make_params_array(1)
    xs, ys = BetaUncertainty.pdf(params)

    # Should return default number of points
    assert len(xs) == BetaUncertainty.default_number_points_in_pdf
    assert len(ys) == BetaUncertainty.default_number_points_in_pdf
    assert len(xs) == len(ys)

    # X values should be in range [0, 1] for standard Beta distribution
    assert np.all(xs >= 0.0)
    assert np.all(xs <= 1.0)

    # Y values should be non-negative
    assert np.all(ys >= 0.0)


def test_pdf_with_scaling(make_params_array):
    """Test PDF calculation with minimum/maximum scaling"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["maximum"] = 4.0

    # Test with specific points
    test_points = np.array([1.5, 2.5, 3.5])
    xs, ys = BetaUncertainty.pdf(params, test_points)

    assert np.allclose(xs, test_points)
    assert len(ys) == 3
    assert np.all(ys >= 0.0)


def test_random_variables_with_scaling(make_params_array):
    """Test random variable generation with scaling"""
    params = make_params_array(1)
    params["minimum"] = 2.0
    params["maximum"] = 5.0

    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (1, 1000)

    # Values should be within the scaled range [2, 5]
    assert np.all(results >= 2.0)
    assert np.all(results <= 5.0)

    # Mean should be approximately the expected scaled mean
    # For alpha=3.3, beta=2.2, mean = 0.6, so scaled mean = 0.6 * 3 + 2 = 3.8
    expected_mean = (ALPHA / (ALPHA + BETA)) * 3.0 + 2.0  # 3.8
    assert 3.6 < np.average(results) < 4.0


def test_cdf_with_scaling(make_params_array):
    """Test CDF calculation with scaling"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["maximum"] = 3.0

    # Test points in the scaled range
    test_points = np.array([[1.2, 2.0, 2.8]])
    cdf_values = BetaUncertainty.cdf(params, test_points)

    assert cdf_values.shape == (1, 3)
    # CDF should be monotonically increasing
    assert cdf_values[0, 0] < cdf_values[0, 1] < cdf_values[0, 2]
    # CDF should be between 0 and 1
    assert np.all(cdf_values >= 0.0)
    assert np.all(cdf_values <= 1.0)


def test_ppf_with_scaling(make_params_array):
    """Test PPF calculation with scaling"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["maximum"] = 3.0

    # Test percentiles
    percentiles = np.array([[0.0, 0.5, 1.0]])
    ppf_values = BetaUncertainty.ppf(params, percentiles)

    assert ppf_values.shape == (1, 3)
    # PPF should return values in the scaled range [1, 3]
    assert np.all(ppf_values >= 1.0)
    assert np.all(ppf_values <= 3.0)
    # PPF should be monotonically increasing
    assert ppf_values[0, 0] < ppf_values[0, 1] < ppf_values[0, 2]


def test_cdf_ppf_roundtrip(make_params_array):
    """Test that CDF and PPF are inverse functions"""
    params = make_params_array(1)
    test_points = np.array([[0.2, 0.5, 0.8]])

    # CDF -> PPF roundtrip
    cdf_values = BetaUncertainty.cdf(params, test_points)
    ppf_values = BetaUncertainty.ppf(params, cdf_values)

    assert np.allclose(test_points, ppf_values, atol=1e-6)


def test_cdf_ppf_roundtrip_with_scaling(make_params_array):
    """Test CDF-PPF roundtrip with scaling"""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["maximum"] = 4.0

    test_points = np.array([[1.5, 2.5, 3.5]])

    # CDF -> PPF roundtrip
    cdf_values = BetaUncertainty.cdf(params, test_points)
    ppf_values = BetaUncertainty.ppf(params, cdf_values)

    assert np.allclose(test_points, ppf_values, atol=1e-6)


def test_validation_min_max_inconsistency(make_params_array):
    """Test validation for min/max inconsistency"""
    params = make_params_array(1)
    params["minimum"] = 5.0
    params["maximum"] = 3.0  # maximum < minimum

    with pytest.raises(ImproperBoundsError):
        BetaUncertainty.validate(params)


def test_validation_min_equals_max(make_params_array):
    """Test validation when minimum equals maximum"""
    params = make_params_array(1)
    params["minimum"] = 3.0
    params["maximum"] = 3.0  # minimum = maximum

    with pytest.raises(ImproperBoundsError):
        BetaUncertainty.validate(params)


def test_validation_nan_values(make_params_array):
    """Test validation with NaN values"""
    params = make_params_array(1)
    params["loc"] = np.nan

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)

    params = make_params_array(1)
    params["shape"] = np.nan

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)


def test_validation_negative_values(make_params_array):
    """Test validation with negative values"""
    params = make_params_array(1)
    params["loc"] = -1.0

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)

    params = make_params_array(1)
    params["shape"] = -2.0

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)


def test_validation_zero_values(make_params_array):
    """Test validation with zero values"""
    params = make_params_array(1)
    params["loc"] = 0.0

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)

    params = make_params_array(1)
    params["shape"] = 0.0

    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)


def test_random_variables_broadcasting_with_scaling(make_params_array):
    """Test random variable generation with broadcasting and scaling"""
    params = make_params_array(2)
    params[0]["minimum"] = 1.0
    params[0]["maximum"] = 3.0
    params[1]["minimum"] = 2.0
    params[1]["maximum"] = 6.0

    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (2, 1000)

    # First row should be in [1, 3]
    assert np.all(results[0, :] >= 1.0)
    assert np.all(results[0, :] <= 3.0)

    # Second row should be in [2, 6]
    assert np.all(results[1, :] >= 2.0)
    assert np.all(results[1, :] <= 6.0)


def test_edge_case_alpha_beta_one(make_params_array):
    """Test edge case where alpha = beta = 1 (uniform distribution)"""
    params = make_params_array(1)
    params["loc"] = 1.0
    params["shape"] = 1.0

    # Should pass validation
    BetaUncertainty.validate(params)

    # Test random variables
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (1, 1000)
    assert np.all(results >= 0.0)
    assert np.all(results <= 1.0)

    # Mean should be approximately 0.5
    assert 0.45 < np.average(results) < 0.55

    # Test statistics
    stats = BetaUncertainty.statistics(params)
    assert np.isclose(stats["mean"], 0.5)
    assert stats["mode"] == "Undefined"  # alpha = beta = 1


def test_edge_case_alpha_beta_large(make_params_array):
    """Test edge case with large alpha and beta values"""
    params = make_params_array(1)
    params["loc"] = 100.0
    params["shape"] = 100.0

    # Should pass validation
    BetaUncertainty.validate(params)

    # Test random variables - should be concentrated around 0.5
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (1, 1000)
    assert np.all(results >= 0.0)
    assert np.all(results <= 1.0)

    # Mean should be very close to 0.5
    assert 0.49 < np.average(results) < 0.51

    # Test statistics
    stats = BetaUncertainty.statistics(params)
    assert np.isclose(stats["mean"], 0.5)
    assert np.isclose(stats["mode"], 0.5)  # Should be defined and equal to mean
