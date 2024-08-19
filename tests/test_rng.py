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
