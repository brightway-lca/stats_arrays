from ...distributions import NormalUncertainty
from ..base import UncertaintyTestCase
import numpy as np
from ...errors import InvalidParamsError


class NormalTestCase(UncertaintyTestCase):

    def test_normal_validate(self):
        params = self.make_params_array(1)
        params['scale'] = np.NaN
        self.assertRaises(
            InvalidParamsError,
            NormalUncertainty.validate,
            params
        )

    def test_normal_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(np.allclose(
            np.array([[1.97475873, 3, 3.673297]]),
            NormalUncertainty.ppf(
                oneDparams,
                np.array([[0.1, 0.5, 0.8]])
            )
        ))
        self.assertTrue(np.allclose(
            np.array([[1.97475873], [3]]),
            NormalUncertainty.ppf(
                params,
                np.array([0.1, 0.5])
            )
        ))

    def test_normal_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(np.allclose(np.array([[0.00620967, 0.10564978, 0.5,
                                               0.89435022]]),
                                    NormalUncertainty.cdf(oneDparams,
                                                          np.array([[1, 2, 3, 4]]))))
        self.assertTrue(np.allclose(np.array([[0.00620967, 0.10564978, 0.5,
                                               0.89435022]]),
                                    NormalUncertainty.cdf(params, np.array([[1, 2, 3, 4],
                                                                           [1, 2, 3, 4]]))[1, :]))

    def test_normal_seeded_random(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertTrue(np.allclose(np.array([[1.49734064]]),
                                    NormalUncertainty.random_variables(oneDparams, 1,
                                                                       self.seeded_random())))

    def test_normal_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        variables = NormalUncertainty.random_variables(oneDparams, size=50000)
        self.assertEqual((1, 50000), variables.shape)
        self.assertTrue(2.95 < np.average(variables) < 3.05)
        self.assertTrue(2.95 < np.median(variables) < 3.05)
        variables = NormalUncertainty.random_variables(params, size=50000)
        self.assertEqual((2, 50000), variables.shape)
        self.assertFalse(np.allclose(variables[0, :], variables[1,:]))
        self.assertTrue(2.95 < np.average(variables[1, :]) < 3.05)
        self.assertTrue(2.95 < np.median(variables[1, :]) < 3.05)
        self.assertTrue(2.95 < np.average(variables[0, :]) < 3.05)

    def test_normal_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertEqual({'upper': 4.5999999046325684,
                          'lower': 1.3999999761581421, 'median': 3.0, 'mode': 3.0,
                          'mean': 3.0},
                         NormalUncertainty.statistics(oneDparams))

    def test_normal_bounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        points = np.array([[1, 2, 3, 4]])
        xs, ys = NormalUncertainty.pdf(oneDparams.copy(), points)
        self.assertTrue(np.allclose(points, xs))
        self.assertTrue(np.allclose(np.array([0.02191038, 0.22831136, 0.49867784,
                                              0.22831136]), ys))
        xs, ys = NormalUncertainty.pdf(oneDparams.copy())
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertEqual(xs.min(), 1)
        self.assertTrue(3.98 < xs.max() <= 4)
        self.assertEqual(ys.min(), 0.021910377331033407)
        self.assertTrue(np.allclose(ys.max(), 0.498668095951))

    def test_normal_unbounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = oneDparams['maximum'] = np.NaN
        xs, ys = NormalUncertainty.pdf(oneDparams.copy())
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertTrue(np.allclose(xs.min(), 1.23999989033))
        self.assertTrue(np.allclose(xs.max(), 4.74241173267))
        self.assertTrue(np.allclose(np.average(xs), 2.9912058115))
        self.assertTrue(np.allclose(ys.min(), 0.0443432302212))
        self.assertTrue(np.allclose(ys.max(), 0.498677843058))
        self.assertTrue(np.allclose(np.average(ys), 0.276188653511))
