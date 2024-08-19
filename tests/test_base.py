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
