import numpy as np
import pytest

from stats_arrays.distributions import (
    BoundedUncertaintyBase,
    NormalUncertainty,
    UncertaintyBase,
    UndefinedUncertainty,
)
from stats_arrays.errors import (
    ImproperBoundsError,
    InvalidParamsError,
    UndefinedDistributionError,
    UnreasonableBoundsError,
)


def make_params_array(length=1):
    assert isinstance(length, int)
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
    return params


def test_uncertainty_base_validate():
    """UncertaintyBase: Mean exists, and bounds are ok if present."""
    params = make_params_array(1)
    params["maximum"] = 2
    params["minimum"] = 2.1
    with pytest.raises(ImproperBoundsError):
        UncertaintyBase.validate(params)


def test_check_2d_inputs():
    params = make_params_array(2)
    params["minimum"] = 0
    params["loc"] = 1
    params["maximum"] = 2
    # Params has 2 rows. The input vector can only have shape (2,) or (2, n)
    with pytest.raises(InvalidParamsError):
        UncertaintyBase.check_2d_inputs(params, np.array((1,)))
    with pytest.raises(InvalidParamsError):
        UncertaintyBase.check_2d_inputs(params, np.array(((1, 2),)))
    with pytest.raises(InvalidParamsError):
        UncertaintyBase.check_2d_inputs(params, np.array(((1, 2), (3, 4), (5, 6))))

    # Test 1-d input
    vector = UncertaintyBase.check_2d_inputs(params, np.array((1, 2)))
    assert np.allclose(vector, np.array(([1], [2])))
    # Test 1-row 2-d input
    vector = UncertaintyBase.check_2d_inputs(params, np.array(((1, 2, 3), (1, 2, 3))))
    assert np.allclose(vector, np.array(((1, 2, 3), (1, 2, 3))))


def test_check_bounds_reasonableness_no_minimum():
    params = make_params_array(1)
    params["maximum"] = -0.3
    params["loc"] = 1
    params["scale"] = 1
    NormalUncertainty.check_bounds_reasonableness(params)


def test_check_bounds_reasonableness():
    params = make_params_array(1)
    params["minimum"] = -100
    params["maximum"] = -0.3
    params["loc"] = 1
    params["scale"] = 1
    with pytest.raises(UnreasonableBoundsError):
        NormalUncertainty.check_bounds_reasonableness(params)


def test_bounded_random_variables():
    params = make_params_array(1)
    params["maximum"] = -0.2  # Only ~ 10 percent of distribution
    params["loc"] = 1
    params["scale"] = 1
    sample = NormalUncertainty.bounded_random_variables(
        params, size=50000, maximum_iterations=1000
    )
    assert (sample > -0.2).sum() == 0
    assert sample.shape == (1, 50000)
    assert np.abs(sample.sum()) > 0


def test_bounded_uncertainty_base_validate():
    """BoundedUncertaintyBase: Make sure legitimate bounds are provided"""
    params = make_params_array(1)
    # Only maximum
    params["maximum"] = 1
    params["minimum"] = np.nan
    with pytest.raises(ImproperBoundsError):
        BoundedUncertaintyBase.validate(params)

    # Only minimum
    params["maximum"] = np.nan
    params["minimum"] = -1
    with pytest.raises(ImproperBoundsError):
        BoundedUncertaintyBase.validate(params)


def test_undefined_uncertainty():
    params = make_params_array(1)
    with pytest.raises(UndefinedDistributionError):
        UndefinedUncertainty.cdf(params, np.random.random(10))
    params = make_params_array(2)
    params["loc"] = 9
    assert np.allclose(
        np.ones((2, 3)) * 9, UndefinedUncertainty.random_variables(params, 3)
    )
    random_percentages = np.random.random(20).reshape(2, 10)
    assert np.allclose(
        np.ones((2, 10)) * 9, UndefinedUncertainty.ppf(params, random_percentages)
    )


def test_rescale_to_unitary_interval_single_row():
    """Test rescale_to_unitary_interval with a single row of parameters."""
    params = make_params_array(1)
    params["minimum"] = 0.0
    params["maximum"] = 10.0
    params["loc"] = 5.0

    adjusted_loc, scale = BoundedUncertaintyBase.rescale_to_unitary_interval(params)

    # Location is at the middle, so should be 0.5 in (0,1) interval
    assert np.allclose(adjusted_loc, 0.5)
    # Scale should be the difference between max and min
    assert np.allclose(scale, 10.0)


def test_rescale_to_unitary_interval_minimum_location():
    """Test rescale_to_unitary_interval when location is at the minimum."""
    params = make_params_array(1)
    params["minimum"] = 2.0
    params["maximum"] = 8.0
    params["loc"] = 2.0

    adjusted_loc, scale = BoundedUncertaintyBase.rescale_to_unitary_interval(params)

    # Location at minimum should map to 0
    assert np.allclose(adjusted_loc, 0.0)
    assert np.allclose(scale, 6.0)


def test_rescale_to_unitary_interval_maximum_location():
    """Test rescale_to_unitary_interval when location is at the maximum."""
    params = make_params_array(1)
    params["minimum"] = 1.0
    params["maximum"] = 5.0
    params["loc"] = 5.0

    adjusted_loc, scale = BoundedUncertaintyBase.rescale_to_unitary_interval(params)

    # Location at maximum should map to 1
    assert np.allclose(adjusted_loc, 1.0)
    assert np.allclose(scale, 4.0)


def test_rescale_to_unitary_interval_multiple_rows():
    """Test rescale_to_unitary_interval with multiple rows of parameters."""
    params = make_params_array(3)
    params["minimum"] = [0.0, 5.0, 10.0]
    params["maximum"] = [10.0, 15.0, 20.0]
    params["loc"] = [5.0, 10.0, 15.0]

    adjusted_loc, scale = BoundedUncertaintyBase.rescale_to_unitary_interval(params)

    # All locations are at the middle
    assert np.allclose(adjusted_loc, [0.5, 0.5, 0.5])
    # Scales should be 10, 10, 10
    assert np.allclose(scale, [10.0, 10.0, 10.0])


def test_rescale_to_unitary_interval_nan_values():
    """Test rescale_to_unitary_interval with NaN values using defaults."""
    params = make_params_array(2)
    params["minimum"][0] = np.nan
    params["minimum"][1] = 5.0
    params["maximum"][0] = np.nan
    params["maximum"][1] = 15.0
    params["loc"] = [2.0, 10.0]

    adjusted_loc, scale = BoundedUncertaintyBase.rescale_to_unitary_interval(params)

    # First row: NaN should default to 0 and 1, location 2 should map to 2
    assert np.allclose(adjusted_loc[0], 2.0)
    assert np.allclose(scale[0], 1.0)
    # Second row: normal case, location in middle
    assert np.allclose(adjusted_loc[1], 0.5)
    assert np.allclose(scale[1], 10.0)


def test_from_tuples_overflow_error():
    """Test that from_tuples raises InvalidParamsError for uncertainty_type >= 256."""
    # Test with uncertainty_type = 256 - should raise error
    with pytest.raises(InvalidParamsError, match="must be less than 256"):
        UncertaintyBase.from_tuples((2, 3, np.nan, np.nan, np.nan, False, 256))

    # Test with valid value (255 should work)
    params = UncertaintyBase.from_tuples((2, 3, np.nan, np.nan, np.nan, False, 255))
    assert params["uncertainty_type"][0] == 255

    # Test with larger value (1000 should raise error)
    with pytest.raises(InvalidParamsError, match="must be less than 256"):
        UncertaintyBase.from_tuples((2, 3, np.nan, np.nan, np.nan, False, 1000))


def test_from_dicts_overflow_error():
    """Test that from_dicts raises InvalidParamsError for uncertainty_type >= 256."""
    # Test with uncertainty_type = 256 - should raise error
    with pytest.raises(InvalidParamsError, match="must be less than 256"):
        UncertaintyBase.from_dicts({"loc": 2, "scale": 3, "uncertainty_type": 256})

    # Test with valid value (255 should work)
    params = UncertaintyBase.from_dicts({"loc": 2, "scale": 3, "uncertainty_type": 255})
    assert params["uncertainty_type"][0] == 255

    # Test with larger value (1000 should raise error)
    with pytest.raises(InvalidParamsError, match="must be less than 256"):
        UncertaintyBase.from_dicts({"loc": 2, "scale": 3, "uncertainty_type": 1000})


def test_from_dicts_uncertainty_type_key():
    """Test from_dicts with 'uncertainty type' (space instead of underscore)."""
    # Should accept both 'uncertainty_type' and 'uncertainty type'
    params1 = UncertaintyBase.from_dicts({"uncertainty_type": 3, "loc": 2, "scale": 3})
    params2 = UncertaintyBase.from_dicts({"uncertainty type": 3, "loc": 2, "scale": 3})

    assert params1["uncertainty_type"][0] == 3
    assert params2["uncertainty_type"][0] == 3


def test_from_dicts_default_values():
    """Test from_dicts with missing values uses defaults."""
    params = UncertaintyBase.from_dicts({"loc": 5})

    assert params["loc"][0] == 5.0
    assert np.isnan(params["scale"][0])
    assert np.isnan(params["shape"][0])
    assert np.isnan(params["minimum"][0])
    assert np.isnan(params["maximum"][0])
    assert params["negative"][0] == False
    assert params["uncertainty_type"][0] == 0


def test_from_dicts_multiple_rows():
    """Test from_dicts with multiple dictionaries."""
    params = UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 3, "uncertainty_type": 3},
        {"loc": 5, "minimum": 3, "maximum": 10, "uncertainty_type": 5},
    )

    assert params.shape[0] == 2
    assert params["loc"][0] == 2.0
    assert params["scale"][0] == 3.0
    assert params["uncertainty_type"][0] == 3
    assert params["loc"][1] == 5.0
    assert params["minimum"][1] == 3.0
    assert params["maximum"][1] == 10.0
    assert params["uncertainty_type"][1] == 5
