import numpy as np
import pytest
from scipy import stats as scipy_stats

from stats_arrays.distributions import DiscreteUniform
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError
from stats_arrays.random import LatinHypercubeRNG


def test_array_shape_1d(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 0
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 100)
    assert sample.shape == (1, 100)


def test_array_shape_2d(make_params_array):
    params = make_params_array(length=10)
    params["minimum"] = 0
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 100)
    assert sample.shape == (10, 100)


def test_random_variables(make_params_array):
    params = make_params_array(length=10)
    params["minimum"] = 5
    params["maximum"] = 10
    sample = DiscreteUniform.random_variables(params, 10000)
    assert np.unique(sample).tolist() == [5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    "minimum, maximum",
    [(5, 10), (5, 9), (0, 2), (-3, 4)],
    ids=["odd count", "even count", "two values", "spans zero"],
)
def test_statistics_match_support(make_params_array, minimum, maximum):
    """Mean and median of the integers minimum..maximum-1.

    Support is [low, high) as in `scipy.stats.randint`, and the mean of the
    integers a..b is (a + b) / 2, so here (minimum + maximum - 1) / 2:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html
    https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    """
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = maximum
    stats = DiscreteUniform.statistics(params)
    support = np.arange(minimum, maximum)
    assert stats["mean"] == scipy_stats.randint(minimum, maximum).mean()
    assert stats["mean"] == support.mean()
    # Midpoint of the two middle values for an even count, as numpy defines it:
    # https://numpy.org/doc/stable/reference/generated/numpy.median.html
    # scipy's `randint.median()` is `ppf(0.5)` and picks the lower one instead.
    assert stats["median"] == np.median(support)
    assert stats["mode"] is None
    assert stats["lower"] == minimum
    assert stats["upper"] == maximum
    assert all(
        isinstance(stats[k], float) for k in ("mean", "median", "lower", "upper")
    )


def test_statistics_mean_matches_samples(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 5
    params["maximum"] = 10
    stats = DiscreteUniform.statistics(params)
    sample = DiscreteUniform.random_variables(params, 100000, np.random.RandomState(0))
    assert stats["mean"] == 7.0
    assert np.isclose(sample.mean(), stats["mean"], atol=0.02)
    assert np.median(sample) == stats["median"]


def test_statistics_nan_minimum_defaults_to_zero(make_params_array):
    params = make_params_array(length=1)
    params["maximum"] = 4
    stats = DiscreteUniform.statistics(params)
    assert stats["lower"] == 0.0
    assert stats["mean"] == stats["median"] == 1.5


def test_cdf(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 2
    params["maximum"] = 6
    result = DiscreteUniform.cdf(params, np.array([[1, 2, 3, 4, 5, 6]]))
    assert np.allclose(result, [[0, 0.25, 0.5, 0.75, 1, 1]])


def test_ppf_stays_on_support(make_params_array):
    params = make_params_array(length=2)
    params["minimum"] = [2, 0]
    params["maximum"] = [6, 3]
    percentages = np.array([[0.01, 0.25, 0.5, 0.99], [0.01, 0.34, 0.67, 0.99]])
    result = DiscreteUniform.ppf(params, percentages)
    assert np.array_equal(result, [[2, 2, 3, 5], [0, 1, 2, 2]])
    assert (DiscreteUniform.cdf(params, result) >= percentages).all()


@pytest.mark.parametrize(
    "minimum, maximum", [(0, 1), (0, 7), (-3, 4), (2, 6), (10, 1010)]
)
def test_ppf_matches_scipy(make_params_array, minimum, maximum):
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = maximum
    count = maximum - minimum
    # Include the steps of the cdf, where rounding matters most
    percentages = np.unique(
        np.concatenate([np.linspace(0.001, 1, 500), np.arange(1, count + 1) / count])
    ).reshape(1, -1)
    expected = scipy_stats.randint.ppf(percentages, minimum, maximum)
    assert np.array_equal(DiscreteUniform.ppf(params, percentages), expected)


def test_ppf_edges(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 2
    params["maximum"] = 6
    result = DiscreteUniform.ppf(params, np.array([[0, 1, -0.1, 1.1, np.nan]]))
    # Always in the support: scipy gives ``minimum - 1`` for 0
    assert np.array_equal(result[0, :2], [2, 5])
    assert np.isnan(result[0, 2:]).all()


@pytest.mark.parametrize("minimum", [2.0**53 - 3, 2.0**52 - 3, -(2.0**53), 1e15])
def test_ppf_large_bounds(make_params_array, minimum):
    """``scipy.stats.randint.ppf`` is biased towards a large minimum."""
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = minimum + 3
    percentages = np.array([[0.01, 0.33, 0.34, 0.4, 0.66, 0.67, 0.7, 0.99, 1]])
    result = DiscreteUniform.ppf(params, percentages)
    assert np.array_equal(result - minimum, [[0, 0, 1, 1, 1, 2, 2, 2, 2]])


def test_latin_hypercube_large_bounds():
    minimum = 2.0**53 - 3
    params = DiscreteUniform.from_dicts(
        {"minimum": minimum, "maximum": 2.0**53, "uncertainty_type": 7}
    )
    hypercube = LatinHypercubeRNG(params, samples=9).hypercube
    assert np.array_equal(hypercube - minimum, [[0, 0, 0, 1, 1, 1, 2, 2, 2]])


def test_latin_hypercube_stays_on_support():
    params = DiscreteUniform.from_dicts(
        {"minimum": 2, "maximum": 6, "uncertainty_type": DiscreteUniform.id}
    )
    hypercube = LatinHypercubeRNG(params, samples=99).hypercube
    values, counts = np.unique(hypercube, return_counts=True)
    assert values.tolist() == [2, 3, 4, 5]
    assert counts.tolist() == [25, 25, 25, 24]


@pytest.mark.parametrize(
    "minimum, maximum",
    # (0.5, 0.9) is the example from #51: both truncate to 0, so sampling failed
    [(0.5, 3), (0, 3.5), (0.5, 3.5), (np.nan, 3.5), (0.5, 0.9)],
)
def test_validate_rejects_non_integer_bounds(make_params_array, minimum, maximum):
    params = make_params_array(length=2)
    params["minimum"] = [0, minimum]
    params["maximum"] = [3, maximum]
    with pytest.raises(InvalidParamsError, match=r"Failing row: \[1\]"):
        DiscreteUniform.validate(params)


@pytest.mark.parametrize(
    "minimum, maximum",
    [
        (0, np.inf),
        (-np.inf, 3),
        (-np.inf, np.inf),
        (np.nan, np.inf),
        (np.nan, -np.inf),
        # These also have minimum >= maximum, but the infinity is reported
        (np.inf, np.inf),
        (-np.inf, -np.inf),
        (np.inf, 3),
    ],
)
def test_validate_rejects_infinite_bounds(make_params_array, minimum, maximum):
    params = make_params_array(length=2)
    params["minimum"] = [0, minimum]
    params["maximum"] = [3, maximum]
    with pytest.raises(
        InvalidParamsError, match=r"must be finite\. Failing row: \[1\]"
    ):
        DiscreteUniform.validate(params)


def test_infinite_bounds_reported_before_bounds_order(make_params_array):
    """Row 0 has minimum >= maximum, but the infinity in row 1 is reported."""
    params = make_params_array(length=2)
    params["minimum"] = [5, 0]
    params["maximum"] = [3, np.inf]
    with pytest.raises(InvalidParamsError, match=r"Failing row: \[1\]"):
        DiscreteUniform.validate(params)


@pytest.mark.parametrize(
    "minimum, maximum",
    [
        (0, 2.0**53 + 2),
        (0, 2.0**63),
        (0, 1e19),
        (0, 1e300),
        (-(2.0**53) - 2, 3),
        (-1e19, 3),
        (-1e300, 3),
        (-1e19, 1e19),
        (np.nan, 1e19),
        # These also have minimum >= maximum, but the range is reported
        (1e19, 3),
        (0, -1e19),
    ],
)
def test_validate_rejects_bounds_beyond_2_53(make_params_array, minimum, maximum):
    params = make_params_array(length=2)
    params["minimum"] = [0, minimum]
    params["maximum"] = [3, maximum]
    with pytest.raises(
        InvalidParamsError, match=r"between -2\*\*53 and 2\*\*53, .* Failing row: \[1\]"
    ):
        DiscreteUniform.validate(params)


def test_validate_accepts_bounds_at_2_53(make_params_array):
    params = make_params_array(length=2)
    params["minimum"] = [-(2.0**53), np.nan]
    params["maximum"] = [2.0**53, 2.0**53]
    DiscreteUniform.validate(params)


@pytest.mark.parametrize(
    "seeded_random", [np.random.default_rng(1), np.random.RandomState(1)]
)
def test_random_variables_at_2_53(make_params_array, seeded_random):
    """Draws next to the limits are exact, so stay below the maximum."""
    params = make_params_array(length=2)
    params["minimum"] = [-(2.0**53), 2.0**53 - 3]
    params["maximum"] = [-(2.0**53) + 3, 2.0**53]
    DiscreteUniform.validate(params)
    sample = DiscreteUniform.random_variables(params, 100, seeded_random)
    assert sample.shape == (2, 100)
    assert set(sample[0]) == {-(2**53), -(2**53) + 1, -(2**53) + 2}
    assert set(sample[1]) == {2**53 - 3, 2**53 - 2, 2**53 - 1}


@pytest.mark.parametrize("rng", ["MCRandomNumberGenerator", "LatinHypercubeRNG"])
def test_rngs_reject_bounds_beyond_2_53(rng):
    import stats_arrays

    params = stats_arrays.UncertaintyBase.from_dicts(
        {"minimum": 0, "maximum": 1e19, "uncertainty_type": 7}
    )
    with pytest.raises(InvalidParamsError, match=r"2\*\*53"):
        getattr(stats_arrays, rng)(params, seed=1)


@pytest.mark.parametrize("rng", ["MCRandomNumberGenerator", "LatinHypercubeRNG"])
def test_rngs_reject_infinite_bounds(rng):
    import stats_arrays

    params = stats_arrays.UncertaintyBase.from_dicts(
        {"minimum": 0, "maximum": np.inf, "uncertainty_type": 7}
    )
    with pytest.raises(InvalidParamsError, match="must be finite"):
        getattr(stats_arrays, rng)(params, seed=1)


@pytest.mark.parametrize("maximum", [0, -2])
def test_validate_rejects_nan_minimum_above_maximum(make_params_array, maximum):
    """#71: a NaN minimum means 0, so the support 0..maximum-1 is empty."""
    params = make_params_array(length=2)
    params["minimum"] = [0, np.nan]
    params["maximum"] = [3, maximum]
    with pytest.raises(
        ImproperBoundsError,
        match=r"minimum >= maximum \(a NaN minimum is read as 0\)\. Failing row: \[1\]",
    ):
        DiscreteUniform.validate(params)


def test_validate_explicit_bounds_error_has_no_nan_hint(make_params_array):
    params = make_params_array(length=2)
    params["minimum"] = [0, 5]
    params["maximum"] = [3, 3]
    with pytest.raises(
        ImproperBoundsError, match=r"minimum >= maximum\. Failing row: \[1\]"
    ):
        DiscreteUniform.validate(params)


def test_validate_does_not_change_nan_minimum(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = np.nan
    params["maximum"] = 3
    DiscreteUniform.validate(params)
    assert np.isnan(params["minimum"][0])


def test_mc_rejects_nan_minimum_above_maximum():
    from stats_arrays import MCRandomNumberGenerator, UncertaintyBase

    params = UncertaintyBase.from_dicts({"maximum": 0, "uncertainty_type": 7})
    with pytest.raises(ImproperBoundsError, match="minimum >= maximum"):
        MCRandomNumberGenerator(params, seed=1)


def test_validate_accepts_nan_minimum_with_maximum_one(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = np.nan
    params["maximum"] = 1
    DiscreteUniform.validate(params)


def test_validate_accepts_nan_minimum(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = np.nan
    params["maximum"] = 3
    DiscreteUniform.validate(params)


@pytest.mark.parametrize(
    "call",
    [
        lambda p: DiscreteUniform.random_variables(p, 5, np.random.default_rng(1)),
        lambda p: DiscreteUniform.random_variables(p, 5, np.random.RandomState(1)),
        lambda p: DiscreteUniform.cdf(p, np.array([1.0, 2.0])),
        lambda p: DiscreteUniform.ppf(p, np.array([[0.5], [0.5]])),
    ],
    ids=["random_variables-Generator", "random_variables-RandomState", "cdf", "ppf"],
)
def test_nan_minimum_not_overwritten(make_params_array, call):
    """Regression test for #52: these methods set a NaN ``minimum`` to 0 in the
    caller's array."""
    params = make_params_array(length=2)
    params["minimum"] = [np.nan, 1]
    params["maximum"] = [3, 4]
    call(params)
    assert np.isnan(params["minimum"][0])
    assert params["minimum"][1] == 1


@pytest.mark.parametrize("method", ["statistics", "pdf"])
def test_nan_minimum_not_overwritten_one_row(make_params_array, method):
    params = make_params_array(length=1)
    params["minimum"] = np.nan
    params["maximum"] = 3
    getattr(DiscreteUniform, method)(params)
    assert np.isnan(params["minimum"][0])


def test_nan_minimum_still_defaults_to_zero(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = np.nan
    params["maximum"] = 3
    data = DiscreteUniform.random_variables(params, 300, np.random.default_rng(1))
    assert set(np.unique(data)) == {0, 1, 2}


def test_mc_keeps_nan_minimum():
    """``MCRandomNumberGenerator`` passes views of its own params, which were
    changed on the first draw."""
    from stats_arrays import MCRandomNumberGenerator, UncertaintyBase

    params = UncertaintyBase.from_dicts(
        {"maximum": 3, "uncertainty_type": DiscreteUniform.id}
    )
    mc = MCRandomNumberGenerator(params, seed=1)
    mc.generate()
    assert np.isnan(mc.params["minimum"][0])
    assert np.isnan(params["minimum"][0])


@pytest.mark.parametrize("minimum, maximum", [(0, 3), (-2, 5), (4, 5), (np.nan, 4)])
def test_pdf_default_xs_is_support(make_params_array, minimum, maximum):
    """Regression test for #60: ``pdf`` raised ``TypeError`` for every input."""
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = maximum
    low = 0 if np.isnan(minimum) else minimum
    xs, ys = DiscreteUniform.pdf(params)
    assert np.array_equal(xs, np.arange(low, maximum))
    assert np.allclose(ys, 1 / (maximum - low))
    assert np.isclose(ys.sum(), 1)


def test_pdf_given_xs(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 0
    params["maximum"] = 3
    xs = np.array([-1, 0, 1.5, 2, 3, 4])
    out_xs, ys = DiscreteUniform.pdf(params, xs)
    assert np.array_equal(out_xs, xs)
    # Outside the support, the excluded maximum, and non-integers: 0
    assert np.allclose(ys, [0, 1 / 3, 0, 1 / 3, 0, 0])


def test_pdf_matches_samples(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = 2
    params["maximum"] = 6
    xs, ys = DiscreteUniform.pdf(params)
    data = DiscreteUniform.random_variables(params, 40_000, np.random.default_rng(1))
    frequencies = np.array([(data == x).mean() for x in xs])
    assert np.allclose(frequencies, ys, atol=0.01)


def test_pdf_default_xs_complete_up_to_cap(make_params_array):
    params = make_params_array(length=1)
    params["minimum"] = -5
    params["maximum"] = DiscreteUniform.max_default_points_in_pdf - 5
    xs, ys = DiscreteUniform.pdf(params)
    assert np.array_equal(
        xs, np.arange(-5, DiscreteUniform.max_default_points_in_pdf - 5)
    )
    assert np.isclose(ys.sum(), 1)


@pytest.mark.parametrize(
    "minimum, maximum",
    [
        (-5, DiscreteUniform.max_default_points_in_pdf - 4),
        (0, 12_345),
        (0, 10**9),
        (2.0**53 - 10**6, 2.0**53),
        (-(2.0**53), 2.0**53),
    ],
)
def test_pdf_default_xs_capped(make_params_array, minimum, maximum):
    """#84: one point per integer of the support raised ``MemoryError``."""
    params = make_params_array(length=1)
    params["minimum"] = minimum
    params["maximum"] = maximum
    xs, ys = DiscreteUniform.pdf(params)
    assert xs.shape == ys.shape
    cap = DiscreteUniform.max_default_points_in_pdf
    assert cap - 1 <= len(xs) <= cap
    assert xs[0] == minimum and xs[-1] == maximum - 1
    # Distinct integers from the support, evenly spaced
    assert (xs == np.floor(xs)).all() and (np.diff(xs) > 0).all()
    steps = np.diff(xs)
    assert steps.max() - steps.min() <= max(1, steps.max() * 1e-6)
    assert np.allclose(ys, 1 / (maximum - minimum))


def test_pdf_cap_can_be_changed(make_params_array, monkeypatch):
    monkeypatch.setattr(DiscreteUniform, "max_default_points_in_pdf", 4)
    params = make_params_array(length=1)
    params["minimum"] = 0
    params["maximum"] = 10
    xs, ys = DiscreteUniform.pdf(params)
    assert np.array_equal(xs, [0, 3, 6, 9])
    assert np.allclose(ys, 0.1)


def test_pdf_given_xs_not_capped(make_params_array, monkeypatch):
    monkeypatch.setattr(DiscreteUniform, "max_default_points_in_pdf", 4)
    params = make_params_array(length=1)
    params["minimum"] = 0
    params["maximum"] = 10
    xs, ys = DiscreteUniform.pdf(params, np.arange(10))
    assert len(xs) == 10 and np.isclose(ys.sum(), 1)
