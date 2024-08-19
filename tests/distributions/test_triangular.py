import pytest
import numpy as np

from stats_arrays.distributions import TriangularUncertainty
from stats_arrays.errors import ImproperBoundsError


def test_bounds(make_params_array):
    params = make_params_array(1)
    params["minimum"] = 1
    params["loc"] = 0
    params["maximum"] = 4
    with pytest.raises(ImproperBoundsError):
        TriangularUncertainty.validate(params)
    params["loc"] = 5
    with pytest.raises(ImproperBoundsError):
        TriangularUncertainty.validate(params)
    params["loc"] = 1
    TriangularUncertainty.validate(params)
    params["loc"] = 4
    TriangularUncertainty.validate(params)


def test_triangular_ppf(biased_params_1d, biased_params_2d):
    percentages = np.array([[0, 0.5, 1]])
    assert np.allclose(
        TriangularUncertainty.ppf(biased_params_1d, percentages),
        np.array([1, 2.73205083, 4]),
    )
    assert np.allclose(
        TriangularUncertainty.ppf(biased_params_2d, np.array([[0, 0.5, 1], [0, 0.5, 1]])),
        np.array([[1, 2.73205083, 4], [1, 2.73205083, 4]]),
    )


def test_triangular_cdf(biased_params_1d, biased_params_2d):
    oneDparams = biased_params_1d
    params = biased_params_2d
    percentages = np.array([0, 0.5, 1])
    assert np.allclose(
        TriangularUncertainty.cdf(oneDparams, np.array([[1, 2.73205083, 4]])),
        percentages,
    )
    assert np.allclose(
        TriangularUncertainty.cdf(params, np.array([1, 2.73205083])),
        np.array([[0], [0.5]]),
    )


def test_triangular_seeded_random(biased_params_1d):
    oneDparams = biased_params_1d
    assert np.allclose(
        2.85968765,
        TriangularUncertainty.random_variables(
            oneDparams, 1, np.random.RandomState(111111)
        ),
    )


def test_triangular_random(biased_params_1d, biased_params_2d):
    oneDparams = biased_params_1d
    params = biased_params_2d
    variables = TriangularUncertainty.random_variables(oneDparams, 5000)
    assert 2.61 < np.average(variables) < 2.71
    assert variables.shape == (1, 5000)
    variables = TriangularUncertainty.random_variables(params, 5000)
    assert not np.allclose(variables[0, :], variables[1, :])
    assert not np.allclose(variables[0, :], variables[1, :])
    assert 2.61 < np.average(variables[0, :]) < 2.71
    assert 2.61 < np.average(variables[1, :]) < 2.71


def test_right_triangles(right_triangle_min, right_triangle_max):
    right_triangle_min_stats = {
        "upper": 3.6648459510835116,
        "lower": 1.0188257132115006,
        "median": 1.8789162245549995,
        "mode": 1.0,
        "mean": 2,
    }
    right_triangle_min_stats_expected_arr = np.array(
        [
            right_triangle_min_stats["upper"],
            right_triangle_min_stats["lower"],
            right_triangle_min_stats["median"],
            right_triangle_min_stats["mode"],
            right_triangle_min_stats["mean"],
        ]
    )
    calculated_stats_min = TriangularUncertainty.statistics(right_triangle_min)
    right_triangle_min_stats_calculated_arr = np.array(
        [
            calculated_stats_min["upper"],
            calculated_stats_min["lower"],
            calculated_stats_min["median"],
            calculated_stats_min["mode"],
            calculated_stats_min["mean"],
        ]
    )
    assert np.allclose(
        right_triangle_min_stats_expected_arr,
        right_triangle_min_stats_calculated_arr,
        rtol=1e-03,
    )

    right_triangle_max_stats = {
        "upper": 3.981087302412936,
        "lower": 1.3350311212247066,
        "median": 3.1208194598959116,
        "mode": 4.0,
        "mean": 3,
    }
    right_triangle_max_stats_expected_arr = np.array(
        [
            right_triangle_max_stats["upper"],
            right_triangle_max_stats["lower"],
            right_triangle_max_stats["median"],
            right_triangle_max_stats["mode"],
            right_triangle_max_stats["mean"],
        ]
    )
    calculated_stats_max = TriangularUncertainty.statistics(right_triangle_max)
    right_triangle_max_stats_calculated_arr = np.array(
        [
            calculated_stats_max["upper"],
            calculated_stats_max["lower"],
            calculated_stats_max["median"],
            calculated_stats_max["mode"],
            calculated_stats_max["mean"],
        ]
    )
    assert np.allclose(
        right_triangle_max_stats_expected_arr,
        right_triangle_max_stats_calculated_arr,
        rtol=1e-03,
    )


def test_triangular_statistics(biased_params_1d):
    tri_stats = {
        "upper": 3.8063508384608244,
        "lower": 1.2738612828334341,
        "median": 2.7320508333784455,
        "mode": 3.0,
        "mean": 2.6666667461395264,
    }
    TriangularUncertainty.statistics(biased_params_1d) == tri_stats


def test_triangular_pdf(biased_params_1d):
    xs, ys = TriangularUncertainty.pdf(biased_params_1d)
    assert np.allclose(np.array([1, 3, 4]), xs)
    assert np.allclose(np.array([0, 0.66666669, 0]), ys)
    points = np.array([1, 2, 3, 4])
    xs, ys = TriangularUncertainty.pdf(biased_params_1d, points)
    assert np.allclose(points, xs)
    assert np.allclose(np.array([0, 0.99999997, 1.99999994, 0]), ys)
