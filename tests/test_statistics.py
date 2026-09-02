"""Every distribution's `statistics` returns scalars, as the base class documents."""

import numpy as np
import pytest

from stats_arrays.distributions import (
    BetaPERTUncertainty,
    LognormalUncertainty,
    UniformUncertainty,
)
from stats_arrays.uncertainty_choices import uncertainty_choices
from stats_arrays.distributions.base import UncertaintyBase

PARAMS = {
    0: {"loc": 1.0},
    1: {"loc": 1.0},
    2: {"loc": 0.5, "scale": 0.2},
    3: {"loc": 0.5, "scale": 0.2},
    4: {"minimum": 1.0, "maximum": 4.0},
    5: {"minimum": 1.0, "maximum": 4.0, "loc": 2.0},
    6: {"loc": 0.3, "minimum": 0.0, "maximum": 1.0},
    7: {"minimum": 1.0, "maximum": 4.0},
    8: {"shape": 1.5, "scale": 2.0},
    9: {"shape": 3.0, "scale": 2.0},
    10: {"loc": 2.0, "shape": 3.0},
    11: {"loc": 1.0, "scale": 2.0, "shape": 0.0},
    12: {"shape": 5.0, "loc": 0.0, "scale": 1.0},
    13: {"minimum": 1.0, "loc": 2.0, "maximum": 4.0},
}

KINDS = list(uncertainty_choices) + [BetaPERTUncertainty]


def one_row(cls):
    return UncertaintyBase.from_dicts({**PARAMS[cls.id], "uncertainty_type": cls.id})


@pytest.mark.parametrize("cls", KINDS, ids=lambda cls: cls.__name__)
def test_statistics_returns_no_arrays(cls):
    """`float(array)` raises on numpy 2, so an array here is a latent error."""
    for key, value in cls.statistics(one_row(cls)).items():
        assert not isinstance(value, np.ndarray), f"{cls.__name__}[{key}]"


@pytest.mark.parametrize("cls", KINDS, ids=lambda cls: cls.__name__)
def test_statistics_are_numbers_or_none(cls):
    """Undefined values are `None`, never a placeholder string."""
    for key, value in cls.statistics(one_row(cls)).items():
        assert isinstance(value, (float, int, type(None))), f"{cls.__name__}[{key}]"


def test_a_lognormal_reports_its_median():
    stats = LognormalUncertainty.statistics(
        UncertaintyBase.from_dicts(
            {"loc": np.log(2.0), "scale": 0.5, "uncertainty_type": 2}
        )
    )
    assert stats["median"] == pytest.approx(2.0)


def test_a_uniform_is_centred_between_its_bounds():
    stats = UniformUncertainty.statistics(
        UncertaintyBase.from_dicts({"minimum": 1.0, "maximum": 4.0, "uncertainty_type": 4})
    )
    assert stats["mean"] == 2.5
    assert stats["lower"] == 1.0
    assert stats["upper"] == 4.0
