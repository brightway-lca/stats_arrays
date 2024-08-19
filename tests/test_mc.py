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
