"""``validate`` rejects infinite distribution parameters (#74)."""

import numpy as np
import pytest

from stats_arrays import (
    LatinHypercubeRNG,
    MCRandomNumberGenerator,
    UncertaintyBase,
    uncertainty_choices,
)
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError

# A valid row for each distribution, and the fields that must be finite
CASES = {
    "normal": ({"loc": 0, "scale": 1, "uncertainty_type": 3}, ["loc", "scale"]),
    "lognormal": ({"loc": 0, "scale": 1, "uncertainty_type": 2}, ["loc", "scale"]),
    "uniform": (
        {"minimum": 0, "maximum": 1, "uncertainty_type": 4},
        ["minimum", "maximum"],
    ),
    "triangular": (
        {"minimum": 0, "loc": 0.5, "maximum": 1, "uncertainty_type": 5},
        ["minimum", "maximum"],
    ),
    "weibull": (
        {"loc": 0, "scale": 1, "shape": 2, "uncertainty_type": 8},
        ["loc", "scale", "shape"],
    ),
    "gamma": (
        {"loc": 0, "scale": 1, "shape": 2, "uncertainty_type": 9},
        ["loc", "scale", "shape"],
    ),
    "beta": (
        {"loc": 2, "shape": 3, "minimum": 0, "maximum": 1, "uncertainty_type": 10},
        ["loc", "shape", "minimum", "maximum"],
    ),
    "gev": (
        {"loc": 0, "scale": 1, "shape": 0, "uncertainty_type": 11},
        ["loc", "scale", "shape"],
    ),
    "students_t": (
        {"loc": 0, "scale": 1, "shape": 3, "uncertainty_type": 12},
        ["loc", "scale", "shape"],
    ),
    "beta_pert": (
        {"minimum": 0, "loc": 0.5, "maximum": 1, "scale": 4, "uncertainty_type": 13},
        ["minimum", "loc", "maximum", "scale"],
    ),
}

# Infinite values that the order checks of the bounds reject first, with the
# ``ImproperBoundsError`` they have always raised
BOUNDED = {4, 5, 10, 13}


def breaks_bounds_order(row, field, value):
    if row["uncertainty_type"] not in BOUNDED:
        return False
    if row["uncertainty_type"] == 13 and field == "loc":
        return True
    return (field, value) in {("minimum", np.inf), ("maximum", -np.inf)}


PARAMETERS = [
    pytest.param(row, field, value, id=f"{name}-{field}-{value}")
    for name, (row, fields) in CASES.items()
    for field in fields
    for value in (np.inf, -np.inf)
]


@pytest.mark.parametrize("name", CASES)
def test_valid_rows_pass(name):
    row, _ = CASES[name]
    params = UncertaintyBase.from_dicts(row, row)
    uncertainty_choices[row["uncertainty_type"]].validate(params)


@pytest.mark.parametrize("row, field, value", PARAMETERS)
def test_validate_rejects_infinite(row, field, value):
    params = UncertaintyBase.from_dicts(row, {**row, field: value})
    distribution = uncertainty_choices[row["uncertainty_type"]]
    if breaks_bounds_order(row, field, value):
        with pytest.raises(ImproperBoundsError, match=r"Failing row: \[1\]"):
            distribution.validate(params)
        return
    with pytest.raises(
        InvalidParamsError,
        match=rf"`{field}` values must not be infinite for {distribution.__name__}\. "
        r"Failing row: \[1\]",
    ):
        distribution.validate(params)


@pytest.mark.parametrize("rng", [MCRandomNumberGenerator, LatinHypercubeRNG])
@pytest.mark.parametrize("uncertainty_type", [4, 10])
def test_infinite_minimum_above_maximum_is_improper_bounds(rng, uncertainty_type):
    """As before #74, so ``except ImproperBoundsError`` still catches it."""
    row = {"loc": 2, "shape": 3, "minimum": np.inf, "maximum": 5}
    params = UncertaintyBase.from_dicts({**row, "uncertainty_type": uncertainty_type})
    with pytest.raises(ImproperBoundsError):
        rng(params, seed=1)


@pytest.mark.parametrize("uncertainty_type", [0, 1])
def test_undefined_and_no_uncertainty_do_not_check_loc(uncertainty_type):
    params = UncertaintyBase.from_dicts(
        {"loc": np.inf, "uncertainty_type": uncertainty_type}
    )
    uncertainty_choices[uncertainty_type].validate(params)
    assert MCRandomNumberGenerator(params, seed=1).generate(2).tolist() == [
        [np.inf, np.inf]
    ]


@pytest.mark.parametrize("rng", [MCRandomNumberGenerator, LatinHypercubeRNG])
@pytest.mark.parametrize(
    "row",
    [
        {"shape": 3, "scale": np.inf, "uncertainty_type": 12},
        {"shape": np.inf, "uncertainty_type": 12},
        {"shape": 2, "scale": 1, "loc": np.inf, "uncertainty_type": 8},
    ],
)
def test_rngs_reject_infinite(rng, row):
    with pytest.raises(InvalidParamsError, match="must not be infinite"):
        rng(UncertaintyBase.from_dicts(row), seed=1)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("uncertainty_type", [2, 3])
def test_optional_bounds_may_be_infinite(uncertainty_type, value):
    """An infinite optional bound means no bound, as NaN does."""
    bound = "minimum" if value < 0 else "maximum"
    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, bound: value, "uncertainty_type": uncertainty_type}
    )
    uncertainty_choices[uncertainty_type].validate(params)


@pytest.mark.parametrize(
    "row",
    [
        {"shape": 3, "uncertainty_type": 12},
        {"shape": 2, "scale": 1, "uncertainty_type": 8},
        {"shape": 2, "scale": 1, "uncertainty_type": 9},
    ],
)
def test_nan_defaults_still_allowed(row):
    """A NaN ``loc`` (or Student's t ``scale``) still means the default."""
    params = UncertaintyBase.from_dicts(row)
    uncertainty_choices[row["uncertainty_type"]].validate(params)
