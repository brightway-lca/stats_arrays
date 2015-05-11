from ...distributions import TriangularUncertainty
from ..base import UncertaintyTestCase
import numpy as np


class TriangularTestCase(UncertaintyTestCase):

    def test_triangular(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()

    def test_triangular_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        percentages = np.array([[0, 0.5, 1]])
        self.assertTrue(np.allclose(TriangularUncertainty.ppf(oneDparams,
                                                              percentages), np.array([1, 2.73205083, 4])))
        self.assertTrue(np.allclose(TriangularUncertainty.ppf(params,
                                                              np.array(
                                                                  [[0, 0.5, 1], [0, 0.5, 1]])), np.array(
            [[1, 2.73205083, 4], [1,
                                  2.73205083, 4]])))

    def test_triangular_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        percentages = np.array([0, 0.5, 1])
        self.assertTrue(np.allclose(TriangularUncertainty.cdf(oneDparams,
                                                              np.array([[1, 2.73205083, 4]])), percentages))
        self.assertTrue(np.allclose(TriangularUncertainty.cdf(params,
                                                              np.array([1, 2.73205083])), np.array([[0], [0.5]])))

    def test_triangular_seeded_random(self):
        oneDparams = self.biased_params_1d()
        self.assertTrue(np.allclose(2.85968765,
                                    TriangularUncertainty.random_variables(oneDparams, 1,
                                                                           self.seeded_random())))

    def test_triangular_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        variables = TriangularUncertainty.random_variables(oneDparams, 5000)
        self.assertTrue(2.61 < np.average(variables) < 2.71)
        self.assertEqual((1, 5000), variables.shape)
        variables = TriangularUncertainty.random_variables(params, 5000)
        self.assertFalse(np.allclose(variables[0, :], variables[1,:]))
        self.assertFalse(np.allclose(variables[0, :], variables[1,:]))
        self.assertTrue(2.61 < np.average(variables[0, :]) < 2.71)
        self.assertTrue(2.61 < np.average(variables[1, :]) < 2.71)

    def test_triangular_statistics(self):
        oneDparams = self.biased_params_1d()
        tri_stats = {'upper': 3.8063508384608244, 'lower': 1.2738612828334341,
                     'median': 2.7320508333784455, 'mode': 3.0,
                     'mean': 2.6666667461395264}
        self.assertEqual(TriangularUncertainty.statistics(oneDparams),
                         tri_stats)

    def test_triangular_pdf(self):
        oneDparams = self.biased_params_1d()
        xs, ys = TriangularUncertainty.pdf(oneDparams)
        self.assertTrue(np.allclose(np.array([1, 3, 4]), xs))
        self.assertTrue(np.allclose(np.array([0, 0.66666669, 0]), ys))
        points = np.array([1, 2, 3, 4])
        xs, ys = TriangularUncertainty.pdf(oneDparams, points)
        self.assertTrue(np.allclose(points, xs))
        self.assertTrue(np.allclose(np.array([0, 0.99999997, 1.99999994, 0]), ys))
