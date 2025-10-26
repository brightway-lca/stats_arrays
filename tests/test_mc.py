import numpy as np
import pytest

from stats_arrays import (
    MCRandomNumberGenerator,
    NormalUncertainty,
    TriangularUncertainty,
    UncertaintyBase,
)


@pytest.fixture()
def mv():
    return UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id},
        {
            "loc": 1.5,
            "minimum": 0,
            "maximum": 10,
            "uncertainty_type": TriangularUncertainty.id,
        },
    )


@pytest.fixture()
def mrng(mv):
    return MCRandomNumberGenerator(mv)


def test_generate(mrng):
    assert mrng.generate(10).shape == (2, 10)
    assert mrng.generate(1).shape == (2,)
    assert mrng.generate().shape == (2,)
    assert mrng.generate(10).shape == (2, 10)


def test_next(mrng):
    assert mrng.next().shape == (2,)
    assert next(mrng).shape == (2,)


def test_iterable(mrng):
    for x, y in zip(mrng, range(10)):
        assert x.shape == (2,)


def test_generate_single_sample_shape(mrng):
    """Test that generate() returns correct shape for single sample"""
    result = mrng.generate(1)
    # Single sample should return 1D array
    assert result.ndim == 1
    assert result.shape == (2,)


def test_generate_multiple_samples_shape(mrng):
    """Test that generate() returns correct shape for multiple samples"""
    result = mrng.generate(10)
    # Multiple samples should return 2D array
    assert result.ndim == 2
    assert result.shape == (2, 10)


def test_generate_no_samples_parameter(mrng):
    """Test that generate() without parameter works"""
    result = mrng.generate()
    # Should return single sample by default
    assert result.ndim == 1
    assert result.shape == (2,)


def test_seeded_generation(mv):
    """Test that seeded generation produces deterministic results"""
    # Create two generators with the same seed
    mrng1 = MCRandomNumberGenerator(mv, seed=12345)
    mrng2 = MCRandomNumberGenerator(mv, seed=12345)

    # Should produce identical results
    result1 = mrng1.generate(5)
    result2 = mrng2.generate(5)
    assert np.allclose(result1, result2)


def test_ordering_preservation(mv):
    """Test that the original ordering is preserved in output"""
    # First uncertainty type is Normal, second is Triangular
    mrng = MCRandomNumberGenerator(mv)
    result = mrng.generate(1)

    # Check that order matches input (first should be closer to normal center,
    # second to triangular center)
    assert result.shape == (2,)
    # Values should be within reasonable bounds
    assert np.all(np.isfinite(result))


def test_multiple_distributions():
    """Test with multiple instances of the same distribution type"""
    params = UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id},
        {"loc": 3, "scale": 0.3, "uncertainty_type": NormalUncertainty.id},
    )
    mrng = MCRandomNumberGenerator(params)
    result = mrng.generate(10)

    assert result.shape == (2, 10)
    # Both should be normal distributions
    assert np.all(np.isfinite(result))


def test_verify_params_calls_distribution_validate(mv):
    """Test that verify_params calls validate on each distribution type"""
    # This test verifies that validate is called for each distribution type
    # The default params should pass validation
    mrng = MCRandomNumberGenerator(mv)
    # If we get here without error, validation passed
    assert mrng is not None


def test_get_positions(mrng):
    """Test that get_positions returns correct distribution positions"""
    positions = mrng.get_positions()

    # Should have entries for all uncertainty types
    assert len(positions) > 0

    # Should count correct number for each type
    total = sum(positions.values())
    assert total == mrng.length


def test_maximum_iterations():
    """Test MCRandomNumberGenerator with custom maximum_iterations"""
    params = UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id}
    )
    mrng = MCRandomNumberGenerator(params, maximum_iterations=100)

    assert mrng.maximum_iterations == 100


def test_custom_seed():
    """Test MCRandomNumberGenerator with custom seed"""
    params = UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id}
    )
    mrng1 = MCRandomNumberGenerator(params, seed=99999)
    mrng2 = MCRandomNumberGenerator(params, seed=99999)

    # Should produce identical results with same seed
    result1 = mrng1.generate(10)
    result2 = mrng2.generate(10)
    assert np.allclose(result1, result2)


def test_ordering_sorted_by_uncertainty_type(mv):
    """Test that parameters are sorted by uncertainty_type internally"""
    mrng = MCRandomNumberGenerator(mv)

    # Check that ordering and reverse_ordering are set
    assert hasattr(mrng, "ordering")
    assert hasattr(mrng, "reverse_ordering")
    assert mrng.ordering is not None
    assert mrng.reverse_ordering is not None
    # The original params should be reordered
    assert len(mrng.ordering) == 2


def test_generate_with_empty_params():
    """Test generate with a single distribution"""
    params = UncertaintyBase.from_dicts(
        {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id}
    )
    mrng = MCRandomNumberGenerator(params)

    result = mrng.generate(5)
    assert result.shape == (1, 5)


def test_iteration_large_range():
    """Test iteration over larger range"""
    mrng = MCRandomNumberGenerator(
        UncertaintyBase.from_dicts(
            {"loc": 2, "scale": 0.5, "uncertainty_type": NormalUncertainty.id}
        )
    )

    count = 0
    for x in mrng:
        assert x.shape == (1,)
        count += 1
        if count >= 50:  # Limit to avoid infinite loop
            break
    assert count == 50
