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
            ("loc", "f4"),
            ("negative", "b1"),
            ("scale", "f4"),
            ("minimum", "f4"),
            ("maximum", "f4"),
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
    lhc = LatinHypercubeRNG(params, seed=11111, samples=9)
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


@pytest.mark.skip()
def test_lognormal_intervals_lower():
    # Lower bound check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    params["loc"] = 1
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


@pytest.mark.skip()
def test_lognormal_intervals_upper():
    # Upper bound check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    params["loc"] = 1
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


@pytest.mark.skip()
def test_lognormal_intervals_both():
    # Both bounds check
    params = make_params_array(1)
    params["scale"] = 1
    params["uncertainty_type"] = LognormalUncertainty.id
    params["loc"] = 1
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


@pytest.mark.skip()
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


@pytest.mark.skip()
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


@pytest.mark.skip()
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
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.6
    lhc = LatinHypercubeRNG(params)
    assert np.allclose(
        lhc.hypercube,
        np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]]),
    )

    # One-dimensional array check
    params = make_params_array(1)
    params["uncertainty_type"] = BernoulliUncertainty.id
    params["maximum"] = 2
    params["minimum"] = 1
    params["loc"] = 1.6
    lhc = LatinHypercubeRNG(params)
    assert np.allclose(lhc.hypercube, np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]))


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
