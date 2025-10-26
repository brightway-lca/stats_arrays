import numpy as np
import pytest

from stats_arrays.distributions import (
    NormalUncertainty,
    TriangularUncertainty,
    UncertaintyBase,
)
from stats_arrays.errors import ImproperBoundsError, UnknownUncertaintyType
from stats_arrays.random import RandomNumberGenerator as RNG


def test_invalid_uncertainty_type():
    with pytest.raises(UnknownUncertaintyType):
        RNG(object, UncertaintyBase.from_dicts({}))


def test_uncertainty_not_subclass():
    class Foo(object):
        @classmethod
        def bounded_random_variables():
            pass

        @classmethod
        def validate(self, *args, **kwargs):
            pass

    RNG(Foo, UncertaintyBase.from_dicts({}))


def test_method_call():
    rng = RNG(NormalUncertainty, UncertaintyBase.from_dicts({"loc": 0, "scale": 1}))
    rng.next()
    rng.generate_random_numbers()


def test_as_iterator():
    counter = 0
    rng = RNG(NormalUncertainty, UncertaintyBase.from_dicts({"loc": 0, "scale": 1}))
    for x in rng:
        counter += 1
        if counter >= 10:
            break


def test_seed():
    data = []
    for x in range(2):
        rng = RNG(
            NormalUncertainty,
            UncertaintyBase.from_dicts({"loc": 0, "scale": 1}),
            seed=111,
        )
        data.append(rng.next())
    assert np.allclose(*data)


def test_output_dimensions():
    rng = RNG(
        NormalUncertainty,
        UncertaintyBase.from_dicts({"loc": 0, "scale": 1}, {"loc": 1, "scale": 2}),
        size=10,
    )
    assert rng.next().shape == (2, 10)


def test_validation():
    with pytest.raises(ImproperBoundsError):
        RNG(
            TriangularUncertainty,
            UncertaintyBase.from_dicts(
                {"loc": -0.000000001, "minimum": 0, "maximum": 1}
            ),
        )

    # No error
    RNG(
        TriangularUncertainty,
        UncertaintyBase.from_dicts({"loc": 0.5, "minimum": 0, "maximum": 1}),
    )
    RNG(
        TriangularUncertainty,
        UncertaintyBase.from_dicts({"loc": 0.0, "minimum": 0, "maximum": 1}),
    )
    RNG(
        TriangularUncertainty,
        UncertaintyBase.from_dicts({"loc": 1.0, "minimum": 0, "maximum": 1}),
    )


@pytest.fixture()
def triangular_params():
    """Create triangular distribution parameters"""
    return TriangularUncertainty.from_dicts(
        {"loc": 5, "minimum": 3, "maximum": 10},
        {"loc": 1, "minimum": 0.7, "maximum": 4.4},
    )


def test_rng_initialization(triangular_params):
    """Test RandomNumberGenerator initialization"""
    rng = RNG(TriangularUncertainty, triangular_params)

    assert rng.params.shape[0] == 2
    assert rng.length == 2
    assert rng.uncertainty_type == TriangularUncertainty
    assert rng.size == 1


def test_rng_with_size(triangular_params):
    """Test RandomNumberGenerator with custom size"""
    rng = RNG(TriangularUncertainty, triangular_params, size=100)

    assert rng.size == 100


def test_rng_generate_random_numbers(triangular_params):
    """Test generate_random_numbers method"""
    rng = RNG(TriangularUncertainty, triangular_params, size=5)
    result = rng.generate_random_numbers()

    assert result.shape == (2, 5)
    # Values should be within triangular bounds
    assert np.all(result >= 0.7)  # Minimum of second parameter
    assert np.all(result <= 10)  # Maximum of first parameter


def test_rng_generate_with_custom_size(triangular_params):
    """Test generate_random_numbers with custom size"""
    rng = RNG(TriangularUncertainty, triangular_params)

    # Use different size
    result = rng.generate_random_numbers(size=10)

    assert result.shape == (2, 10)


def test_rng_verify_params_with_invalid_bounds():
    """Test verify_params validation"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params)

    # Try to verify with invalid parameters
    invalid_params = TriangularUncertainty.from_dicts(
        {"loc": 10, "minimum": 5, "maximum": 1}  # Invalid: min > max
    )

    with pytest.raises(ImproperBoundsError):
        rng.verify_params(params=invalid_params, uncertainty_type=TriangularUncertainty)


def test_verify_params_defaults():
    """Test verify_params with default parameters"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params)

    # Calling verify_params without args should use defaults
    rng.verify_params()


def test_verify_uncertainty_type_default():
    """Test verify_uncertainty_type with default"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params)

    # Should have the correct uncertainty_type
    assert rng.uncertainty_type == TriangularUncertainty


def test_generate_random_numbers_with_custom_params():
    """Test generate_random_numbers with custom uncertainty_type and params"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params)

    # Use different params but same uncertainty_type
    other_params = TriangularUncertainty.from_dicts(
        {"loc": 3, "minimum": 2, "maximum": 6}
    )
    result = rng.generate_random_numbers(
        uncertainty_type=TriangularUncertainty, params=other_params
    )

    assert result.shape == (1, 1)  # size=1 is the default


def test_generate_random_numbers_different_types():
    """Test generate_random_numbers with different uncertainty types"""
    # Create NormalUncertainty
    params = UncertaintyBase.from_dicts({"loc": 2, "scale": 0.5})
    rng = RNG(NormalUncertainty, params)

    result = rng.generate_random_numbers()
    assert result.shape == (1, 1)
    assert np.all(np.isfinite(result))


def test_maximum_iterations():
    """Test RandomNumberGenerator with custom maximum_iterations"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params, maximum_iterations=200)

    assert rng.maximum_iterations == 200


def test_random_initialization():
    """Test that random state is initialized"""
    params = NormalUncertainty.from_dicts({"loc": 0, "scale": 1})
    rng = RNG(NormalUncertainty, params, seed=42)

    assert rng.random is not None
    assert isinstance(rng.random, np.random.RandomState)


def test_params_property():
    """Test that params property is accessible"""
    params = TriangularUncertainty.from_dicts({"loc": 2, "minimum": 1, "maximum": 5})
    rng = RNG(TriangularUncertainty, params)

    assert rng.params is not None
    assert rng.params.shape[0] == 1


def test_length_property():
    """Test length property"""
    params = NormalUncertainty.from_dicts(
        {"loc": 2, "scale": 0.5}, {"loc": 3, "scale": 0.3}
    )
    rng = RNG(NormalUncertainty, params)

    assert rng.length == 2


def test_iteration_with_multiple_samples():
    """Test iteration when size > 1"""
    params = NormalUncertainty.from_dicts({"loc": 2, "scale": 0.5})
    rng = RNG(NormalUncertainty, params, size=10)

    result = rng.next()
    assert result.shape == (1, 10)
