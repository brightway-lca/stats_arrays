"""Every distribution's `statistics` returns scalars, as the base class documents."""

import re

import numpy as np
import pytest

from stats_arrays.distributions import (
    BetaPERTUncertainty,
    LognormalUncertainty,
    UniformUncertainty,
)
from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError, MultipleRowParamsArrayError
from stats_arrays.uncertainty_choices import uncertainty_choices

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
        UncertaintyBase.from_dicts(
            {"minimum": 1.0, "maximum": 4.0, "uncertainty_type": 4}
        )
    )
    assert stats["mean"] == 2.5
    assert stats["lower"] == 1.0
    assert stats["upper"] == 4.0


def two_rows(cls):
    row = {**PARAMS[cls.id], "uncertainty_type": cls.id}
    return UncertaintyBase.from_dicts(row, row)


@pytest.mark.parametrize("method", ["statistics", "pdf", "check_bounds_reasonableness"])
@pytest.mark.parametrize("cls", KINDS, ids=lambda cls: cls.__name__)
def test_one_row_methods_reject_two_rows(cls, method):
    """#65: a 1-D params array of length 2 holds two distributions."""
    with pytest.raises(MultipleRowParamsArrayError, match=r"got 2 distributions"):
        getattr(cls, method)(two_rows(cls))


@pytest.mark.parametrize(
    "shape, count", [((2,), 2), ((2, 1), 2), ((1, 2), 2), ((2, 3), 6)]
)
def test_one_row_rejects_every_multi_row_shape(shape, count):
    row = {**PARAMS[LognormalUncertainty.id], "uncertainty_type": 2}
    params = UncertaintyBase.from_dicts(*[row] * count).reshape(shape)
    message = (
        "statistics needs a params array with exactly one row, "
        f"but got {count} distributions (shape {shape})"
    )
    with pytest.raises(MultipleRowParamsArrayError, match=re.escape(message)):
        LognormalUncertainty.statistics(params)


@pytest.mark.parametrize("shape", [(0,), (0, 1), (1, 0)])
def test_one_row_rejects_empty_params(shape):
    """No rows is not "multiple rows", so it gets a different error."""
    params = one_row(LognormalUncertainty)[:0].reshape(shape)
    message = f"got an empty array (shape {shape})"
    with pytest.raises(InvalidParamsError, match=re.escape(message)):
        LognormalUncertainty.statistics(params)


@pytest.mark.parametrize(
    "reshape",
    [
        lambda params: params,
        lambda params: params.reshape(1, 1),
        lambda params: params.reshape(()),
        lambda params: params[0],
    ],
    ids=["(1,)", "(1, 1)", "0-d array", "numpy.void scalar"],
)
def test_one_row_accepts_one_row_shapes(reshape):
    params = reshape(
        UncertaintyBase.from_dicts(
            {"loc": np.log(2.0), "scale": 0.5, "uncertainty_type": 2}
        )
    )
    stats = LognormalUncertainty.statistics(params)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["mean"] == pytest.approx(2.0 * np.exp(0.125))
    assert stats["lower"] == pytest.approx(2.0 / np.e)
    assert stats["upper"] == pytest.approx(2.0 * np.e)
    xs, ys = LognormalUncertainty.pdf(params)
    assert xs.shape == ys.shape == (LognormalUncertainty.default_number_points_in_pdf,)
