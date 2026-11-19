import numpy as np
import pytest

from stats_arrays.distributions import (
    BernoulliUncertainty,
    LognormalUncertainty,
    NormalUncertainty,
    NoUncertainty,
    TriangularUncertainty,
    UndefinedUncertainty,
    UniformUncertainty,
)
from stats_arrays.random import LatinHypercubeRNG


def make_params_array(length=1):
    assert isinstance(length, int)
    params = np.zeros(
        (length,),
        dtype=[
            ("uncertainty_type", "i2"),
            ("input", "u4"),
            ("output", "u4"),
            ("loc", "f8"),
            ("negative", "b1"),
            ("scale", "f8"),
            ("minimum", "f8"),
            ("maximum", "f8"),
        ],
    )
    params["minimum"] = params["maximum"] = params["scale"] = np.nan
    return params


def test_known_seed():
    params = make_params_array(1)
    params["uncertainty_type"] = UniformUncertainty.id
    params["maximum"] = 1
    params["minimum"] = 0
    params["loc"] = 0.5
    params["scale"] = np.nan
    # Legacy RandomState stream, as in stats_arrays < 5
    lhc = LatinHypercubeRNG(params, seed=np.random.RandomState(11111), samples=9)
    assert np.allclose(next(lhc), np.array([[0.5]]))
    assert np.allclose(next(lhc), np.array([[0.7]]))
    assert np.allclose(next(lhc), np.array([[0.5]]))
    assert np.allclose(next(lhc), np.array([[0.6]]))


def test_no_uncertainty_intervals():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = NoUncertainty.id
    params["loc"] = (1, 2)
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    )
    assert np.allclose(lhc.hypercube, test_values)
    params["uncertainty_type"] = UndefinedUncertainty.id
    lhc = LatinHypercubeRNG(params)
    assert np.allclose(lhc.hypercube, test_values)
    # One-dimensional array check
    params = make_params_array(1)
    params["loc"] = 1
    params["uncertainty_type"] = NoUncertainty.id
    lhc = LatinHypercubeRNG(params)
    test_values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_lognormal_intervals_2d():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = LognormalUncertainty.id
    params["loc"] = (1, 2)
    params["scale"] = (1, 2)
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                0.71521093,
                1.09586291,
                1.48499978,
                1.91792577,
                2.42495914,
                3.0470846,
                3.85262859,
                4.97579579,
                6.74268247,
                10.3312957,
            ],
            [
                0.51152672,
                1.20091561,
                2.20522451,
                3.67843954,
                5.88042728,
                9.28472525,
                14.84274814,
                24.75854564,
                45.4637703,
                106.7356788,
            ],
        ]
    )
    assert np.allclose(lhc.hypercube, test_values)


def test_lognormal_intervals_1d():
    # One-dimensional array check
    params = make_params_array(1)
    params["loc"] = 1
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            0.71521093,
            1.09586291,
            1.48499978,
            1.91792577,
            2.42495914,
            3.0470846,
            3.85262859,
            4.97579579,
            6.74268247,
            10.3312957,
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_lognormal_intervals_lower():
    # Lower bound check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    # Expected values are for mu = 0, i.e. a median of exp(0) = 1
    params["loc"] = 0
    params["minimum"] = 1
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                1.12095981,
                1.25845417,
                1.41730289,
                1.604463,
                1.83049303,
                2.11247151,
                2.48049433,
                2.99457872,
                3.8006714,
                5.42285066,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_lognormal_intervals_upper():
    # Upper bound check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    # Expected values are for mu = 0, i.e. a median of exp(0) = 1
    params["loc"] = 0
    params["maximum"] = 1
    params["minimum"] = np.nan
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                0.18440486,
                0.26311141,
                0.33393679,
                0.40314545,
                0.47337916,
                0.54630091,
                0.62326149,
                0.70556548,
                0.79462568,
                0.89209264,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_lognormal_intervals_both():
    # Both bounds check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    # Expected values are for mu = 0, i.e. a median of exp(0) = 1
    params["loc"] = 0
    params["minimum"] = 0.5
    params["maximum"] = 1.5
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                0.56110473,
                0.62514369,
                0.69285366,
                0.76502388,
                0.84254518,
                0.92646272,
                1.01804071,
                1.11884934,
                1.23088804,
                1.35676875,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_normal_intervals_2d():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = NormalUncertainty.id
    params["loc"] = (1, 2)
    params["scale"] = (1, 2)
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        (
            [
                -0.33517774,
                0.09154213,
                0.39541465,
                0.6512443,
                0.88581471,
                1.11418529,
                1.3487557,
                1.60458535,
                1.90845787,
                2.33517774,
            ],
            [
                -0.67035547,
                0.18308426,
                0.79082931,
                1.30248861,
                1.77162941,
                2.22837059,
                2.69751139,
                3.20917069,
                3.81691574,
                4.67035547,
            ],
        )
    )
    assert np.allclose(lhc.hypercube, test_values)


def test_normal_intervals_1d():
    # One-dimensional array check
    params = make_params_array(1)
    params["loc"] = 1
    params["scale"] = 1
    params["uncertainty_type"] = NormalUncertainty.id
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            -0.33517774,
            0.09154213,
            0.39541465,
            0.6512443,
            0.88581471,
            1.11418529,
            1.3487557,
            1.60458535,
            1.90845787,
            2.33517774,
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_normal_intervals_lower():
    # Lower bound check
    params = make_params_array(1)
    params["loc"] = 1
    params["scale"] = 1
    params["uncertainty_type"] = NormalUncertainty.id
    params["minimum"] = 1
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                1.11418529,
                1.22988412,
                1.3487557,
                1.47278912,
                1.60458535,
                1.74785859,
                1.90845787,
                2.09680356,
                2.33517774,
                2.69062163,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_normal_intervals_upper():
    # Upper bound check
    params = make_params_array(1)
    params["loc"] = 1
    params["scale"] = 1
    params["uncertainty_type"] = NormalUncertainty.id
    params["maximum"] = 1
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                -0.69062163,
                -0.33517774,
                -0.09680356,
                0.09154213,
                0.25214141,
                0.39541465,
                0.52721088,
                0.6512443,
                0.77011588,
                0.88581471,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_normal_intervals_both():
    # Both bounds check
    params = make_params_array(1)
    params["loc"] = 1
    params["scale"] = 1
    params["uncertainty_type"] = NormalUncertainty.id
    params["minimum"] = 0.5
    params["maximum"] = 1.5
    lhc = LatinHypercubeRNG(params)
    test_values = np.array(
        [
            [
                0.59665956,
                0.68968379,
                0.78009253,
                0.86873532,
                0.95635658,
                1.04364342,
                1.13126468,
                1.21990747,
                1.31031621,
                1.40334044,
            ]
        ]
    ).reshape(1, 10)
    assert np.allclose(lhc.hypercube, test_values)


def test_uniform_intervals():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = UniformUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.5
    params["scale"] = np.nan
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(
        lhc.hypercube,
        np.array(
            [
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            ]
        ),
    )

    # One-dimensional array check
    params = make_params_array(1)
    params["uncertainty_type"] = UniformUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.5
    params["scale"] = np.nan
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(
        lhc.hypercube, np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])
    )


def test_triangular_intervals():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = TriangularUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.6
    params["scale"] = np.nan
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(
        lhc.hypercube,
        np.array(
            [
                [
                    1.24494898,
                    1.34641017,
                    1.42426408,
                    1.48989796,
                    1.54772257,
                    1.60000001,
                    1.65358985,
                    1.7171573,
                    1.80000001,
                ],
                [
                    1.24494898,
                    1.34641017,
                    1.42426408,
                    1.48989796,
                    1.54772257,
                    1.60000001,
                    1.65358985,
                    1.7171573,
                    1.80000001,
                ],
            ]
        ),
    )

    # One-dimensional array check
    params = make_params_array(1)
    params["uncertainty_type"] = TriangularUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.6
    params["scale"] = np.nan
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(
        lhc.hypercube,
        np.array(
            [
                [
                    1.24494898,
                    1.34641017,
                    1.42426408,
                    1.48989796,
                    1.54772257,
                    1.60000001,
                    1.65358985,
                    1.7171573,
                    1.80000001,
                ]
            ]
        ),
    )


def test_bernoulli():
    # Two-dimensional array check
    params = make_params_array(2)
    params["uncertainty_type"] = BernoulliUncertainty.id
    params["loc"] = 0.6
    lhc = LatinHypercubeRNG(params)
    print(lhc.hypercube)
    assert np.allclose(
        lhc.hypercube,
        np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ]
        ),
    )

    # One-dimensional array check
    params = make_params_array(1)
    params["uncertainty_type"] = BernoulliUncertainty.id
    params["loc"] = 0.4
    lhc = LatinHypercubeRNG(params)
    assert np.allclose(lhc.hypercube, np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))


def test_heterogeneous_params():
    params = make_params_array(2)
    params["uncertainty_type"][0] = TriangularUncertainty.id
    params["uncertainty_type"][1] = UniformUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.6
    params["scale"] = np.nan
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(
        lhc.hypercube,
        np.array(
            [
                [
                    1.24494898,
                    1.34641017,
                    1.42426408,
                    1.48989796,
                    1.54772257,
                    1.60000001,
                    1.65358985,
                    1.7171573,
                    1.80000001,
                ],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            ]
        ),
    )


def test_lh_with_only_minimum_bound():
    """Test LatinHypercubeRNG with only minimum bound (no maximum)"""
    # Use Normal which doesn't require both bounds
    params = make_params_array(1)
    params["uncertainty_type"] = NormalUncertainty.id
    params["loc"] = 2
    params["scale"] = 0.5
    params["minimum"] = 1.0
    params["maximum"] = np.nan

    lhc = LatinHypercubeRNG(params, samples=20)

    assert lhc.hypercube.shape == (1, 20)
    # Should still produce finite values
    assert np.all(np.isfinite(lhc.hypercube))


def test_lh_with_only_maximum_bound():
    """Test LatinHypercubeRNG with only maximum bound (no minimum)"""
    # Use Normal which doesn't require both bounds
    params = make_params_array(1)
    params["uncertainty_type"] = NormalUncertainty.id
    params["loc"] = 2
    params["scale"] = 0.5
    params["minimum"] = np.nan
    params["maximum"] = 5.0

    lhc = LatinHypercubeRNG(params, samples=20)

    assert lhc.hypercube.shape == (1, 20)
    # Should still produce finite values
    assert np.all(np.isfinite(lhc.hypercube))


def test_lh_with_unbounded_distribution():
    """Test LatinHypercubeRNG with unbounded distribution (no min, no max)"""
    # Use Normal which doesn't require bounds
    params = make_params_array(1)
    params["uncertainty_type"] = NormalUncertainty.id
    params["loc"] = 2
    params["scale"] = 0.5
    params["minimum"] = np.nan
    params["maximum"] = np.nan

    lhc = LatinHypercubeRNG(params, samples=20)

    assert lhc.hypercube.shape == (1, 20)
    assert np.all(np.isfinite(lhc.hypercube))


def test_lh_with_large_samples():
    """Test LatinHypercubeRNG with large number of samples"""
    params = make_params_array(1)
    params["uncertainty_type"] = NormalUncertainty.id
    params["loc"] = 2
    params["scale"] = 0.5

    lhc = LatinHypercubeRNG(params, samples=100)

    assert lhc.hypercube.shape == (1, 100)
    assert np.all(np.isfinite(lhc.hypercube))


def test_lh_all_distributions():
    """Test LatinHypercubeRNG with all supported distributions"""
    from stats_arrays.distributions import UniformUncertainty

    # Test with uniform
    params = make_params_array(1)
    params["uncertainty_type"] = UniformUncertainty.id
    params["minimum"] = 1.0
    params["maximum"] = 2.0
    params["loc"] = 1.5
    params["scale"] = np.nan

    lhc = LatinHypercubeRNG(params, samples=10)
    assert lhc.hypercube.shape == (1, 10)
    assert np.all(lhc.hypercube >= 1.0)
    assert np.all(lhc.hypercube <= 2.0)


def lhc_bounds_params(*rows):
    from stats_arrays import UncertaintyBase

    return UncertaintyBase.from_dicts(*rows)


@pytest.mark.parametrize(
    "bounds",
    [{"minimum": 0.5}, {"maximum": 0.5}, {"minimum": 0.5, "maximum": 1.5}],
)
def test_bounds_respected_in_hypercube(bounds):
    """Regression test for #48, with the example from the issue."""
    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id, **bounds}
    )
    lhc = LatinHypercubeRNG(params, samples=9, seed=1)
    assert (lhc.hypercube >= bounds.get("minimum", -np.inf)).all()
    assert (lhc.hypercube <= bounds.get("maximum", np.inf)).all()
    # Still Latin hypercube: strictly increasing, no repeated values
    assert (np.diff(lhc.hypercube[0]) > 0).all()


def test_bounds_mixed_rows():
    """Bounded and unbounded rows of the same and different types."""
    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"loc": 0, "scale": 1, "minimum": 0, "uncertainty_type": NormalUncertainty.id},
        {
            "loc": 0,
            "scale": 1,
            "maximum": 1,
            "uncertainty_type": LognormalUncertainty.id,
        },
        {
            "loc": 0,
            "scale": 1,
            "minimum": -1,
            "maximum": 1,
            "uncertainty_type": NormalUncertainty.id,
        },
    )
    lhc = LatinHypercubeRNG(params, samples=20)
    unbounded = NormalUncertainty.ppf(params[:1], lhc.percentages[:1])
    assert np.allclose(lhc.hypercube[0], unbounded)
    assert (lhc.hypercube[0] < 0).any()
    assert (lhc.hypercube[1] >= 0).all()
    assert ((lhc.hypercube[2] > 0) & (lhc.hypercube[2] <= 1)).all()
    assert ((lhc.hypercube[3] >= -1) & (lhc.hypercube[3] <= 1)).all()


def test_bounds_negative_lognormal():
    """For a lognormal with ``negative`` set, ``ppf`` decreases, so the signed
    minimum limits the upper percentage and the signed maximum the lower one."""
    params = lhc_bounds_params(
        {
            "loc": 0,
            "scale": 1,
            "negative": True,
            "minimum": -2,
            "maximum": -0.5,
            "uncertainty_type": LognormalUncertainty.id,
        }
    )
    lhc = LatinHypercubeRNG(params, samples=10)
    assert ((lhc.hypercube >= -2) & (lhc.hypercube <= -0.5)).all()


@pytest.mark.parametrize(
    "row",
    [
        {"minimum": 3, "maximum": 6, "uncertainty_type": 7},  # DiscreteUniform
        {"minimum": 1, "maximum": 2, "uncertainty_type": UniformUncertainty.id},
        {
            "loc": 1.5,
            "minimum": 1,
            "maximum": 2,
            "uncertainty_type": TriangularUncertainty.id,
        },
    ],
)
def test_bounds_that_are_parameters_change_nothing(row):
    """Where ``minimum`` and ``maximum`` define the support, the hypercube must
    cover the whole distribution, as it did before the fix."""
    params = lhc_bounds_params(row)
    lhc = LatinHypercubeRNG(params, samples=9)
    assert np.allclose(lhc.percentages, np.arange(1, 10).reshape(1, -1) / 10)


def test_bounds_discrete_uniform_samples_minimum():
    params = lhc_bounds_params({"minimum": 3, "maximum": 6, "uncertainty_type": 7})
    lhc = LatinHypercubeRNG(params, samples=9)
    assert set(np.unique(lhc.hypercube)) == {3, 4, 5}


@pytest.mark.parametrize(
    "uncertainty_type", [UndefinedUncertainty.id, NoUncertainty.id]
)
def test_bounds_ignored_for_undefined(uncertainty_type):
    """These used to raise ``UndefinedDistributionError`` when bounds were set."""
    params = lhc_bounds_params(
        {"loc": 2, "minimum": 1, "maximum": 3, "uncertainty_type": uncertainty_type}
    )
    lhc = LatinHypercubeRNG(params, samples=4)
    assert (lhc.hypercube == 2).all()


@pytest.mark.parametrize("uncertainty_type", [NormalUncertainty, LognormalUncertainty])
@pytest.mark.parametrize("bounds", [(2, 1), (1, 1)], ids=["reversed", "equal"])
def test_minimum_not_below_maximum_rejected(uncertainty_type, bounds):
    """Normal and lognormal ``validate`` skipped the base-class bounds check, so
    reversed bounds gave a reversed hypercube, and equal bounds a constant one."""
    from stats_arrays import ImproperBoundsError, MCRandomNumberGenerator

    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": uncertainty_type.id},
        {
            "loc": 0,
            "scale": 1,
            "minimum": bounds[0],
            "maximum": bounds[1],
            "uncertainty_type": uncertainty_type.id,
        },
    )
    with pytest.raises(ImproperBoundsError, match=r"Failing row: \[1\]"):
        uncertainty_type.validate(params)
    with pytest.raises(ImproperBoundsError):
        LatinHypercubeRNG(params)
    with pytest.raises(ImproperBoundsError):
        MCRandomNumberGenerator(params)


@pytest.mark.parametrize("uncertainty_type", [NormalUncertainty, LognormalUncertainty])
@pytest.mark.parametrize(
    "bounds", [{}, {"minimum": 0.5}, {"maximum": 2}, {"minimum": 0.5, "maximum": 2}]
)
def test_ordered_or_missing_bounds_still_validate(uncertainty_type, bounds):
    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": uncertainty_type.id, **bounds}
    )
    uncertainty_type.validate(params)


@pytest.mark.parametrize(
    "row",
    [
        {"loc": 0, "scale": 1, "minimum": 9, "uncertainty_type": NormalUncertainty.id},
        {
            "loc": 0,
            "scale": 1,
            "maximum": -40,
            "uncertainty_type": NormalUncertainty.id,
        },
        {
            # Positive bounds on a negative lognormal: outside its support
            "loc": 0,
            "scale": 0.3,
            "negative": True,
            "minimum": 0.5,
            "maximum": 2,
            "uncertainty_type": LognormalUncertainty.id,
        },
    ],
    ids=["far-upper-tail", "far-lower-tail", "outside-support"],
)
def test_bounds_without_probability_mass_rejected(row):
    """These gave a hypercube of all ``inf``, ``-inf`` or 0."""
    from stats_arrays import UnreasonableBoundsError

    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id}, row
    )
    with pytest.raises(UnreasonableBoundsError, match=r"Failing row: \[1\]"):
        LatinHypercubeRNG(params, samples=5)


@pytest.mark.parametrize(
    "row",
    [
        {"loc": 0, "scale": 1, "minimum": 5, "uncertainty_type": NormalUncertainty.id},
        {"loc": 0, "scale": 1, "maximum": -8, "uncertainty_type": NormalUncertainty.id},
        {"minimum": 4, "maximum": 5, "uncertainty_type": 7},  # DiscreteUniform
        {"loc": 2, "minimum": 5, "maximum": 6, "uncertainty_type": NoUncertainty.id},
    ],
    ids=["upper-tail", "lower-tail", "single-value-support", "no-uncertainty"],
)
def test_thin_but_usable_bounds_accepted(row):
    lhc = LatinHypercubeRNG(lhc_bounds_params(row), samples=5)
    assert np.isfinite(lhc.hypercube).all()


@pytest.mark.parametrize(
    "name, row",
    [
        (
            "WeibullUncertainty",
            {"loc": 0, "scale": 1, "shape": 2, "uncertainty_type": 8},
        ),
        ("GammaUncertainty", {"loc": 0, "scale": 1, "shape": 2, "uncertainty_type": 9}),
        (
            "GeneralizedExtremeValueUncertainty",
            {"loc": 0, "scale": 1, "shape": 0, "uncertainty_type": 11},
        ),
        (
            "StudentsTUncertainty",
            {"loc": 0, "scale": 1, "shape": 3, "uncertainty_type": 12},
        ),
    ],
)
def test_distribution_without_ppf_rejected_clearly(name, row):
    """These raised ``NotImplementedError`` with no message."""
    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id}, row
    )
    with pytest.raises(NotImplementedError) as excinfo:
        LatinHypercubeRNG(params)
    message = str(excinfo.value)
    assert name in message
    assert "`ppf`" in message
    assert "Failing row: [1]" in message


def test_unsupported_distributions_all_listed():
    params = lhc_bounds_params(
        {"loc": 0, "scale": 1, "shape": 2, "uncertainty_type": 9},
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"loc": 0, "scale": 1, "shape": 3, "uncertainty_type": 12},
    )
    with pytest.raises(NotImplementedError) as excinfo:
        LatinHypercubeRNG(params)
    message = str(excinfo.value)
    assert "GammaUncertainty" in message and "StudentsTUncertainty" in message
    assert "Failing rows: [0, 2]" in message


def _choices_with(monkeypatch, distribution):
    from stats_arrays.uncertainty_choices import UncertaintyChoices

    choices = UncertaintyChoices()
    choices.add(distribution)
    monkeypatch.setattr("stats_arrays.random.uncertainty_choices", choices)


def test_distribution_without_cdf_only_rejected_for_bounded_rows(monkeypatch):
    from stats_arrays import UncertaintyBase

    class PpfOnly(UncertaintyBase):
        id = 77
        description = "Has ppf but no cdf"

        @classmethod
        def ppf(cls, params, percentages):
            return np.zeros(percentages.shape) + params["loc"].reshape(-1, 1)

    _choices_with(monkeypatch, PpfOnly)
    unbounded = lhc_bounds_params({"loc": 3, "uncertainty_type": 77})
    assert (LatinHypercubeRNG(unbounded, samples=4).hypercube == 3).all()

    mixed = lhc_bounds_params(
        {"loc": 3, "uncertainty_type": 77},
        {"loc": 3, "minimum": 1, "uncertainty_type": 77},
    )
    with pytest.raises(NotImplementedError) as excinfo:
        LatinHypercubeRNG(mixed, samples=4)
    message = str(excinfo.value)
    assert "PpfOnly" in message and "`cdf`" in message and "`ppf`;" not in message
    assert "Failing row: [1]" in message


def test_staticmethod_ppf_counts_as_implemented(monkeypatch):
    from stats_arrays import UncertaintyBase

    class StaticPpf(UncertaintyBase):
        id = 78
        description = "ppf is a staticmethod"

        @staticmethod
        def ppf(params, percentages):
            return np.zeros(percentages.shape) + 5

    _choices_with(monkeypatch, StaticPpf)
    params = lhc_bounds_params({"loc": 3, "uncertainty_type": 78})
    assert (LatinHypercubeRNG(params, samples=4).hypercube == 5).all()


@pytest.mark.parametrize("samples", [1, 2, 5])
def test_negative_lognormal_is_negative(samples):
    """Regression test for #54: negative lognormals came out positive.

    ``samples=1`` takes the single-column branch of
    ``LognormalUncertainty.ppf``, which negates separately."""
    params = lhc_bounds_params(
        {"loc": 0, "scale": 0.3, "uncertainty_type": LognormalUncertainty.id},
        {
            "loc": 0,
            "scale": 0.3,
            "negative": True,
            "uncertainty_type": LognormalUncertainty.id,
        },
    )
    lhc = LatinHypercubeRNG(params, samples=samples)
    assert lhc.hypercube.shape == (2, samples)
    assert (lhc.hypercube[1] < 0).all()
    assert np.allclose(lhc.hypercube[1], -lhc.hypercube[0])


@pytest.mark.parametrize(
    "row",
    [
        {"loc": -5, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"minimum": -3, "maximum": -1, "uncertainty_type": UniformUncertainty.id},
        {
            "loc": -2,
            "minimum": -3,
            "maximum": -1,
            "uncertainty_type": TriangularUncertainty.id,
        },
    ],
)
def test_negative_flag_ignored_where_monte_carlo_ignores_it(row):
    """Regression test for #54: rows with ``negative`` set were flipped, while
    ``MCRandomNumberGenerator`` leaves them alone."""
    plain = LatinHypercubeRNG(lhc_bounds_params(row), samples=5).hypercube
    flagged = LatinHypercubeRNG(
        lhc_bounds_params({**row, "negative": True}), samples=5
    ).hypercube
    assert np.array_equal(plain, flagged)
    assert (flagged < 0).all()


@pytest.mark.parametrize(
    "row",
    [
        {"loc": 0, "scale": 0.3, "uncertainty_type": LognormalUncertainty.id},
        {"loc": -5, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"loc": 1, "scale": 0.5, "uncertainty_type": LognormalUncertainty.id},
    ],
)
def test_negative_sign_matches_monte_carlo(row):
    from stats_arrays import MCRandomNumberGenerator

    params = lhc_bounds_params({**row, "negative": True})
    lhc = LatinHypercubeRNG(params, samples=20).hypercube
    mc = MCRandomNumberGenerator(params, seed=1).generate(samples=200)
    assert np.sign(np.median(lhc)) == np.sign(np.median(mc))


@pytest.fixture()
def lhc_two_rows():
    return lhc_bounds_params(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"minimum": 0, "maximum": 1, "uncertainty_type": UniformUncertainty.id},
    )


def test_generate_single_sample(lhc_two_rows):
    """Regression test for #50: ``generate()`` raised ``AttributeError``."""
    lhc = LatinHypercubeRNG(lhc_two_rows, seed=1, samples=5)
    data = lhc.generate()
    assert data.shape == (2,)
    for row in range(2):
        assert data[row] in lhc.hypercube[row]


def test_generate_matches_next(lhc_two_rows):
    a = LatinHypercubeRNG(lhc_two_rows, seed=3, samples=5)
    b = LatinHypercubeRNG(lhc_two_rows, seed=3, samples=5)
    for _ in range(5):
        assert np.array_equal(a.generate(), next(b))


def test_generate_multiple_samples(lhc_two_rows):
    lhc = LatinHypercubeRNG(lhc_two_rows, seed=1, samples=5)
    data = lhc.generate(samples=200)
    assert data.shape == (2, 200)
    for row in range(2):
        assert set(data[row]) == set(lhc.hypercube[row])


def test_generate_out_and_order(lhc_two_rows):
    expected = LatinHypercubeRNG(lhc_two_rows, seed=2, samples=5).generate(samples=3)
    lhc = LatinHypercubeRNG(lhc_two_rows, seed=2, samples=5)
    out = np.full((2, 3), np.nan)
    assert lhc.generate(samples=3, out=out, order="sorted") is out
    assert np.array_equal(out, expected)
    assert np.array_equal(lhc.ordering, np.arange(2))


def test_generate_argument_checks(lhc_two_rows):
    lhc = LatinHypercubeRNG(lhc_two_rows, samples=5)
    with pytest.raises(ValueError):
        lhc.generate(order="random")
    with pytest.raises(ValueError):
        lhc.generate(out=np.empty(3))
    with pytest.raises(TypeError):
        lhc.generate(out=np.empty(2, dtype=int))
