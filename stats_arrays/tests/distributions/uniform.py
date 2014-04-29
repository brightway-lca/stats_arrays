from ...distributions import UniformUncertainty
from ..base import UncertaintyTestCase
import numpy as np


class UniformTestCase(UncertaintyTestCase):

    def unif_params_1d(self):
        oneDparams = self.make_params_array(1)
        oneDparams['minimum'] = 1
        oneDparams['loc'] = 2
        oneDparams['maximum'] = 3
        return oneDparams

    def unif_params_2d(self):
        params = self.make_params_array(2)
        params['minimum'] = 1
        params['loc'] = 2
        params['maximum'] = 3
        return params

    def test_uniform_ppf(self):
        oneDparams = self.unif_params_1d()
        params = self.unif_params_2d()
        self.assertTrue(np.allclose(np.array([1, 2, 3]),
                                    UniformUncertainty.ppf(oneDparams, np.array([[0, 0.5, 1]]))))
        self.assertTrue(np.allclose(np.array([[1], [2]]),
                                    UniformUncertainty.ppf(params, np.array([0, 0.5]))))

    def test_uniform_cdf(self):
        oneDparams = self.unif_params_1d()
        params = self.unif_params_2d()
        self.assertTrue(np.allclose(np.array([0, 0.5, 1]),
                                    UniformUncertainty.cdf(oneDparams, np.array([[1, 2, 3, ]]))))
        self.assertTrue(np.allclose(np.array([[0], [0.5]]),
                                    UniformUncertainty.cdf(params, np.array([1, 2]))))

    def test_uniform_seeded_random(self):
        oneDparams = self.unif_params_1d()
        self.assertTrue(np.allclose(2.15281272,
                                    UniformUncertainty.random_variables(oneDparams, 1,
                                                                        self.seeded_random())))

    def test_uniform_random(self):
        oneDparams = self.unif_params_1d()
        params = self.unif_params_2d()
        variables = UniformUncertainty.random_variables(oneDparams, size=5000)
        self.assertEqual((1, 5000), variables.shape)
        self.assertTrue(1.95 < np.average(variables) < 2.05)
        variables = UniformUncertainty.random_variables(params, size=5000)
        self.assertEqual((2, 5000), variables.shape)
        self.assertFalse(np.allclose(variables[0, :], variables[1,:]))
        self.assertTrue(1.95 < np.average(variables[0, :]) < 2.05)
        self.assertTrue(1.95 < np.average(variables[1, :]) < 2.05)

    def test_uniform_statistics(self):
        oneDparams = self.unif_params_1d()
        self.assertEqual({'mean': 2, 'mode': 2, 'median': 2, 'lower': 1,
                          'upper': 3}, UniformUncertainty.statistics(oneDparams))

    def test_uniform_pdf(self):
        oneDparams = self.unif_params_1d()
        xs, ys = UniformUncertainty.pdf(oneDparams)
        self.assertTrue(np.allclose(np.array([1, 3]), xs))
        self.assertTrue(np.allclose(np.array([0.5, 0.5]), ys))
        points = np.array([1, 2, 3])
        xs, ys = UniformUncertainty.pdf(oneDparams, points)
        self.assertTrue(np.allclose(points, xs))
        self.assertTrue(np.allclose(np.array([0.5, 0.5, 0.5]), ys))
