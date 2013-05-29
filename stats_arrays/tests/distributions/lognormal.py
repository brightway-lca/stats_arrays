from ...errors import ImproperBoundsError, \
    InvalidParamsError
from ...distributions import LognormalUncertainty
from ..base import UncertaintyTestCase
import numpy as np


class LognormalTestCase(UncertaintyTestCase):
    def pdf(self, x, mu, sigma):
        return 1 / (x * np.sqrt(2 * np.pi * sigma ** 2)
            ) * np.e ** (-((np.log(x) - mu) ** 2) / (
            2 * sigma ** 2))

    def test_lognormal_implementation(self):
        """
Test lognormal implementation against known and calculated values.

Every time I re-learn how SciPy does lognormal I get confused again. Imagine a lognormal distribution with a geometric mean of 0.5, an underlying mean of log(0.5) (this is the mean of the underlying normal distribution), and a scale (standard deviation) of 0.5.

The formula for PDF is::

    1/(x*sqrt(2*pi*scale**2))*e**(-((log(x)-mu)**2)/(2*scale**2))

The formula for CDF is::

    0.5 + 0.5*erf((log(x) - mu)/sqrt(2*pi*scale**2))

Known values: The geometric mean is also the median.

* CDF(x=0.5, mu=log(0.5), sigma=1) = 0.5
* PPF(x=0.5, mu=log(0.5), sigma=1) = 0.5
* PDF(x=1, mu=log(1), sigma=1) = 1/sqrt(2*pi) = 0.398942280401
* PDF(x=0.8, mu=log(0.7), sigma=0.6) = 1/(0.8*sqrt(2*pi*0.36))*e**(-((log(0.8/0.7))**2)/0.72) = 0.81079978795919017

        """
        params = self.make_params_array()
        params['loc'] = 0.5
        params['scale'] = 1.
        self.assertTrue(np.allclose(
            LognormalUncertainty.ppf(
                params.copy(),
                np.array([[0.5]])),
            0.5
        ))
        self.assertTrue(np.allclose(
            LognormalUncertainty.cdf(
                params.copy(),
                np.array([[0.5]])),
            0.5
        ))
        params['loc'] = 1.
        self.assertTrue(np.allclose(
            LognormalUncertainty.pdf(
                params.copy(),
                np.array([[1.]]))[1],
            1 / np.sqrt(2 * np.pi)
        ))

    def test_lognormal_set_negative_flag(self):
        params = self.biased_params_1d()
        LognormalUncertainty.set_negative_flag(params)
        self.assertFalse(params['negative'])
        params['loc'] = -1.
        LognormalUncertainty.set_negative_flag(params)
        self.assertTrue(params['negative'])
        self.assertEqual(params['loc'][0], 1.)

        params = self.biased_params_2d()
        LognormalUncertainty.set_negative_flag(params)
        self.assertEquals(params['negative'], 0)
        params['loc'][0] = -1.
        LognormalUncertainty.set_negative_flag(params)
        self.assertEquals(params['negative'], 1)
        self.assertEqual(params['loc'][0], 1.)

    def test_lognormal_validate_required_values(self):
        params = self.make_params_array(1)
        params['loc'] = 1
        self.assertRaises(InvalidParamsError, LognormalUncertainty.validate,
            params, transform=True)
        params['scale'] = 0.8
        params['loc'] = np.NaN
        self.assertRaises(InvalidParamsError, LognormalUncertainty.validate,
            params, transform=True)
        params['scale'] = -0.8
        params['loc'] = 1
        self.assertRaises(InvalidParamsError, LognormalUncertainty.validate,
            params, transform=True)

    def test_lognormal_validate_minimum_maximum(self):
        params = self.make_params_array(1)
        params['maximum'] = 2
        params['loc'] = 1
        params['minimum'] = 2.1
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)
        params = self.make_params_array(1)
        params['maximum'] = -2
        params['loc'] = -1.75
        params['minimum'] = -1.5
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)

    def test_lognormal_validate_mean_out_of_bounds(self):
        # Mean > maximum
        params = self.make_params_array(1)
        params['maximum'] = 2
        params['loc'] = 2.1
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)
        params = self.make_params_array(1)
        params['maximum'] = -2
        params['loc'] = -1
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)
        # Mean < minimum
        params = self.make_params_array(1)
        params['minimum'] = 2
        params['loc'] = 1.9
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)
        params = self.make_params_array(1)
        params['minimum'] = -2
        params['loc'] = -3
        params['scale'] = 0.8
        self.assertRaises(ImproperBoundsError, LognormalUncertainty.validate,
            params, transform=True)

    def test_lognormal_ppf(self):
        params = self.biased_params_1d()
        params['scale'] = 0.8
        params['loc'] = 2
        self.assertTrue(np.allclose(
            np.array([[2]]),
            LognormalUncertainty.ppf(params, np.array([0.5]))
        ))

    def test_lognormal_negative_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        params['scale'] = 0.8
        params['minimum'] = -4
        params['loc'] = -3
        params['maximum'] = -1
        self.assertTrue(np.allclose(np.array([[-0.83281874,-3.00000006,-6.96037632]]),
            LognormalUncertainty.ppf(oneDparams, np.array([[0.1, 0.5, 0.8]]),
            transform=True)))
        self.assertTrue(np.allclose(np.array([[-0.83281874], [-3.00000006]]),
            LognormalUncertainty.ppf(params, np.array([0.1, 0.5]),
            transform=True)))

    def test_lognormal_negative_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        params['scale'] = 0.8
        params['minimum'] = -4
        params['loc'] = -3
        params['maximum'] = -1
        self.assertTrue(np.allclose(np.array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(oneDparams.copy(),
            -1*np.array([[1,2,3,4]]), transform=True)))
        self.assertTrue(np.allclose(np.array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(params.copy(),
            -1*np.array([[1,2,3,4], [1,2,3,4]]), transform=True)[1,:]))

    def test_lognormal_seeded_random(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertTrue(np.allclose(np.array([[0.66761271]]),
            LognormalUncertainty.random_variables(oneDparams, 1,
            self.seeded_random(), transform=True)))

    def test_lognormal_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        variables = LognormalUncertainty.random_variables(
            oneDparams,
            size=50000
        )
        self.assertEqual((1, 50000), variables.shape)
        self.assertTrue(4 < np.average(variables) < 4.25)
        self.assertTrue(2.9 < np.median(variables) < 3.1)
        variables = LognormalUncertainty.random_variables(params, size=50000)
        self.assertEqual((2, 50000), variables.shape)
        self.assertFalse(np.allclose(variables[0, :], variables[1, :]))
        self.assertTrue(4 < np.average(variables[0, :]) < 4.25)
        self.assertTrue(2.9 < np.median(variables[1, :]) < 3.1)
        self.assertTrue(4 < np.average(variables[1, :]) < 4.25)

    def test_lognormal_negative_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        params['scale'] = 0.8
        params['minimum'] = -4
        params['loc'] = -3
        params['maximum'] = -1
        variables = LognormalUncertainty.random_variables(oneDparams,
            size=50000, transform=True)
        self.assertEqual((1, 50000), variables.shape)
        self.assertTrue(4 < -1 * np.average(variables) < 4.25)
        self.assertTrue(2.9 < -1 * np.median(variables) < 3.1)
        variables = LognormalUncertainty.random_variables(params, size=50000,
            transform=True)
        self.assertEqual((2,50000), variables.shape)
        self.assertFalse(np.allclose(variables[0,:], variables[1,:]))
        self.assertTrue(4 < -1 * np.average(variables[0,:]) < 4.25)
        self.assertTrue(2.9 < -1 * np.median(variables[1,:]) < 3.1)
        self.assertTrue(4 < -1 * np.average(variables[1,:]) < 4.25)

    def test_lognormal_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        to_array = lambda x: np.array([x[key] for key in sorted(x.keys())])
        self.assertTrue(np.allclose(
            to_array({
                'upper': 14.859097922170479,
                'lower': 0.60568955155650162,
                'median': 3.0000000595022631,
                'mode': 1.5818772733323232,
                'mean': 4.131383414350033
            }),
            to_array(LognormalUncertainty.statistics(oneDparams))
        ))

    def test_lognormal_negative_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        to_array = lambda x: np.array([x[key] for key in sorted(x.keys())])
        self.assertTrue(np.allclose(
            to_array({
                'lower': -14.859097922170479,
                'upper': -0.60568955155650162,
                'median': -3.0000000595022631,
                'mode': -1.5818772733323232,
                'mean': -4.131383414350033
            }),
            to_array(LognormalUncertainty.statistics(oneDparams, transform=True))
        ))

    def test_lognormal_bounded_pdf(self):
        params = self.biased_params_1d()
        params['scale'] = 0.8
        points = np.array([[1, 2, 3, 4]])
        xs, ys = LognormalUncertainty.pdf(
            params.copy(),
            points
        )
        self.assertTrue(np.allclose(points, xs))
        self.assertEqual(xs.shape, (4,))
        self.assertEqual(ys.shape, (4,))
        self.assertTrue(np.allclose(ys, np.array([self.pdf(x, np.log(3), 0.8) for x \
            in points])))
        xs, ys = LognormalUncertainty.pdf(
            params.copy(),
            transform=True
        )
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertEqual(xs.min(), 1)
        self.assertTrue(3.97 < xs.max() <= 4)
        stats = LognormalUncertainty.statistics(params, transform=True)
        self.assertTrue(abs(ys.min() - min(self.pdf(1, log(3), 0.8),
            self.pdf(4, log(3), 0.8))) / ys.min() < 0.01)
        self.assertTrue(abs(ys.max() - self.pdf(stats['mode'], log(3), 0.8)
            ) / ys.max() < 0.01)
        self.assertTrue(np.allclose(ys[-1], self.pdf(xs[-1], log(3), 0.8)))
        self.assertTrue(np.allclose(ys[0], self.pdf(xs[0], log(3), 0.8)))
