"""``cdf`` and ``ppf`` take one row of values for all rows of params (#75)."""

import numpy as np
import pytest

from stats_arrays import UncertaintyBase, uncertainty_choices
from stats_arrays.distributions import LognormalUncertainty

# Two different rows for each distribution with ``cdf`` and ``ppf``
ROWS = {
    "no_uncertainty": [{"loc": 2}, {"loc": 3}],
    "lognormal": [{"loc": 0, "scale": 1}, {"loc": 1, "scale": 0.5, "negative": True}],
    "normal": [{"loc": 0, "scale": 1}, {"loc": 1, "scale": 2}],
    "uniform": [{"minimum": 0, "maximum": 1}, {"minimum": -2, "maximum": 4}],
    "triangular": [
        {"minimum": 0, "loc": 0.5, "maximum": 1},
        {"minimum": -2, "loc": 3, "maximum": 4},
    ],
    "bernoulli": [{"loc": 0.3}, {"loc": 0.8}],
    "discrete_uniform": [{"minimum": 0, "maximum": 5}, {"minimum": -3, "maximum": 2}],
    "weibull": [{"scale": 1, "shape": 2}, {"loc": 1, "scale": 2, "shape": 1.5}],
    "gamma": [{"scale": 1, "shape": 2}, {"loc": 1, "scale": 3, "shape": 0.7}],
    "beta": [
        {"loc": 2, "shape": 3},
        {"loc": 1, "shape": 1.5, "minimum": -2, "maximum": 4},
    ],
    "gev": [{"loc": 0, "scale": 1, "shape": 0}, {"loc": -3, "scale": 4, "shape": 0}],
    "student": [{"shape": 3}, {"loc": 2, "scale": 0.5, "shape": 10}],
    "beta_pert": [
        {"minimum": 0, "loc": 0.5, "maximum": 1},
        {"minimum": -2, "loc": 3, "maximum": 4},
    ],
}
TYPES = {
    "no_uncertainty": 1,
    "lognormal": 2,
    "normal": 3,
    "uniform": 4,
    "triangular": 5,
    "bernoulli": 6,
    "discrete_uniform": 7,
    "weibull": 8,
    "gamma": 9,
    "beta": 10,
    "gev": 11,
    "student": 12,
    "beta_pert": 13,
}
VALUES = np.array([[-1.0, 0.0, 0.25, 0.5, 1.0, 2.5]])
PERCENTAGES = np.array([[0.01, 0.2, 0.5, 0.8, 0.99]])


def make(name):
    rows = [{**row, "uncertainty_type": TYPES[name]} for row in ROWS[name]]
    return uncertainty_choices[TYPES[name]], UncertaintyBase.from_dicts(*rows)


def test_all_distributions_with_ppf_covered():
    covered = set(TYPES.values())
    assert {d.id for d in uncertainty_choices if d.id != 0} == covered


@pytest.mark.parametrize("name", [n for n in ROWS if n != "no_uncertainty"])
def test_cdf_one_row(name):
    distribution, params = make(name)
    expected = distribution.cdf(params, np.repeat(VALUES, 2, axis=0))
    result = distribution.cdf(params, VALUES.copy())
    assert result.shape == (2, VALUES.shape[1])
    assert np.allclose(result, expected, equal_nan=True)
    # The rows differ, so each row of params is used
    assert not np.allclose(result[0], result[1], equal_nan=True)


@pytest.mark.parametrize("name", ROWS)
def test_ppf_one_row(name):
    distribution, params = make(name)
    expected = distribution.ppf(params, np.repeat(PERCENTAGES, 2, axis=0))
    result = distribution.ppf(params, PERCENTAGES.copy())
    assert result.shape == (2, PERCENTAGES.shape[1])
    assert np.allclose(result, expected)
    assert not np.allclose(result[0], result[1])


@pytest.mark.parametrize(
    "name, method",
    [(name, "ppf") for name in ROWS]
    + [(name, "cdf") for name in ROWS if name != "no_uncertainty"],
)
def test_wrong_number_of_rows(name, method):
    from stats_arrays.errors import InvalidParamsError

    distribution, params = make(name)
    with pytest.raises(InvalidParamsError, match="params has 2 rows"):
        getattr(distribution, method)(params, np.ones((3, 2)) / 2)


def test_lognormal_negative_one_row():
    """Each row's sign applies, and the shared row is not changed."""
    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "uncertainty_type": 2},
        {"loc": 0, "scale": 1, "negative": True, "uncertainty_type": 2},
    )
    vector = np.array([[-2.0, -1.0, 1.0, 2.0]])
    result = LognormalUncertainty.cdf(params, vector)
    assert np.allclose(vector, [[-2.0, -1.0, 1.0, 2.0]])
    assert np.allclose(result[0], LognormalUncertainty.cdf(params[:1], vector)[0])
    assert np.allclose(result[1], LognormalUncertainty.cdf(params[:1], -vector)[0])
    ppf = LognormalUncertainty.ppf(params, PERCENTAGES)
    assert np.allclose(ppf[1], -ppf[0])


@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 3)])
def test_lognormal_cdf_does_not_change_vector(shape):
    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "negative": True, "uncertainty_type": 2},
        {"loc": 0, "scale": 1, "negative": True, "uncertainty_type": 2},
    )
    vector = -np.ones(shape)
    LognormalUncertainty.cdf(params, vector)
    assert (vector == -1).all()


def test_lognormal_cdf_checks_shape_first():
    from stats_arrays.errors import InvalidParamsError

    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "uncertainty_type": 2},
        {"loc": 0, "scale": 1, "uncertainty_type": 2},
    )
    with pytest.raises(InvalidParamsError):
        LognormalUncertainty.cdf(params, np.ones((3, 2)))


@pytest.mark.parametrize("negative", [False, True])
def test_lognormal_params_with_shape_1_1(negative):
    """Params for one distribution can be 2-dimensional, as
    ``one_row_params_array`` passes them."""
    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "negative": negative, "uncertainty_type": 2}
    )
    sign = -1 if negative else 1
    for method, vector in [("cdf", sign * VALUES[:, 2:]), ("ppf", PERCENTAGES)]:
        expected = getattr(LognormalUncertainty, method)(params, vector)
        result = getattr(LognormalUncertainty, method)(params.reshape(1, 1), vector)
        assert result.shape == vector.shape
        assert np.allclose(result, expected)
    assert (
        sign * LognormalUncertainty.ppf(params.reshape(1, 1), PERCENTAGES) > 0
    ).all()


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float32])
def test_lognormal_cdf_integer_vector(dtype):
    """An unsigned vector raised ``OverflowError`` for rows with ``negative``."""
    params = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "uncertainty_type": 2},
        {"loc": 0, "scale": 1, "negative": True, "uncertainty_type": 2},
    )
    vector = np.array([1, 2], dtype=dtype)
    result = LognormalUncertainty.cdf(params, vector)
    assert np.allclose(result[0], LognormalUncertainty.cdf(params[:1], np.array([1.0])))
    # A negative lognormal has no mass above zero
    assert result[1, 0] == 0
    assert vector.dtype == dtype and vector.tolist() == [1, 2]


def test_check_2d_inputs_returns_floats():
    params = UncertaintyBase.from_dicts({"loc": 1}, {"loc": 2})
    for vector in [np.array([1, 2], dtype=np.uint8), np.array([[1, 2, 3]])]:
        assert UncertaintyBase.check_2d_inputs(params, vector).dtype == float


@pytest.mark.parametrize("uncertainty_type", [0, 1])
def test_undefined_ppf_checks_rows(uncertainty_type):
    from stats_arrays.errors import InvalidParamsError

    distribution = uncertainty_choices[uncertainty_type]
    params = UncertaintyBase.from_dicts({"loc": 1}, {"loc": 2})
    assert distribution.ppf(params, np.zeros((1, 3))).tolist() == [[1] * 3, [2] * 3]
    with pytest.raises(InvalidParamsError):
        distribution.ppf(params, np.zeros((5, 3)))


@pytest.mark.parametrize("name", [n for n in ROWS if n != "no_uncertainty"])
@pytest.mark.parametrize("negative", [False, True])
def test_params_with_shape_1_1(name, negative):
    """Gamma and Weibull ``ppf`` raised ``IndexError`` too."""
    row = {**ROWS[name][0], "negative": negative, "uncertainty_type": TYPES[name]}
    params = UncertaintyBase.from_dicts(row)
    distribution = uncertainty_choices[TYPES[name]]
    for method, vector in [("cdf", VALUES), ("ppf", PERCENTAGES)]:
        expected = getattr(distribution, method)(params, vector)
        result = getattr(distribution, method)(params.reshape(1, 1), vector)
        assert np.allclose(result, expected, equal_nan=True)
