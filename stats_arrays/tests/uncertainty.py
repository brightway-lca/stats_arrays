import unittest
from numpy import *
from scipy.special import erf
from ..distributions import *
from ..errors import ImproperBoundsError, \
    UndefinedDistributionError, InvalidParamsError, UnreasonableBoundsError
from nose.plugins.skip import SkipTest


class UncertaintyTestCase(unittest.TestCase):
    def make_params_array(self, length=1):
        assert isinstance(length, int)
        params = zeros((length,), dtype=[('input', 'u4'), ('output', 'u4'),
            ('loc', 'f4'), ('negative', 'b1'), ('scale', 'f4'),
            ('minimum', 'f4'), ('maximum', 'f4')])
        params['minimum'] = params['maximum'] = params['scale'] = NaN
        return params

    def seeded_random(self, seed=111111):
        return random.RandomState(seed)

    def biased_params_1d(self):
        oneDparams = self.make_params_array(1)
        oneDparams['minimum'] = 1
        oneDparams['loc'] = 3
        oneDparams['maximum'] = 4
        return oneDparams

    def biased_params_2d(self):
        params = self.make_params_array(2)
        params['minimum'] = 1
        params['loc'] = 3
        params['maximum'] = 4
        return params

    def test_uncertainty_base_validate(self):
        """UncertaintyBase: Mean exists, and bounds are ok if present."""
        params = self.make_params_array(1)
        params['maximum'] = 2
        params['loc'] = 1.6
        # Minimum too big
        params['minimum'] = 1.8
        self.assertRaises(ImproperBoundsError, UncertaintyBase.validate,
            params)
        # Mean above max
        params['minimum'] = 1
        params['loc'] = 2.5
        self.assertRaises(ImproperBoundsError, UncertaintyBase.validate,
            params)
        # Mean below min
        params['loc'] = 0.5
        self.assertRaises(ImproperBoundsError, UncertaintyBase.validate,
            params)
        # No mean
        params['loc'] = NaN
        self.assertRaises(InvalidParamsError, UncertaintyBase.validate,
            params)

    # def test_random_timing(self):
    #     import time
    #     t = time.time()
    #     params = self.make_params_array(1)
    #     params['loc'] = 1
    #     params['scale'] = 1
    #     sample = NormalUncertainty.random_variables(params, size=50000)
    #     print "Without limits: %.4f" % (time.time() - t)
    #     t = time.time()
    #     params = self.make_params_array(1)
    #     params['loc'] = 1
    #     params['scale'] = 1
    #     sample = NormalUncertainty.bounded_random_variables(params, size=50000)
    #     print "Without limits, but with bounded_r_v: %.4f" % (time.time() - t)
    #     t = time.time()
    #     params = self.make_params_array(1)
    #     params['maximum'] = -0.2
    #     params['loc'] = 1
    #     params['scale'] = 1
    #     sample = NormalUncertainty.bounded_random_variables(params, size=50000, maximum_iterations=1000)
    #     print "With limits: %.4f" % (time.time() - t)

    def test_check_2d_inputs(self):
        params = self.make_params_array(2)
        params['minimum'] = 0
        params['loc'] = 1
        params['maximum'] = 2
        # Params has 2 rows. The input vector can only have shape (2,) or (2, n)
        self.assertRaises(InvalidParamsError, UncertaintyBase.check_2d_inputs,
            params, array((1,)))
        self.assertRaises(InvalidParamsError, UncertaintyBase.check_2d_inputs,
            params, array(((1, 2),)))
        self.assertRaises(InvalidParamsError, UncertaintyBase.check_2d_inputs,
            params, array(((1, 2), (3, 4), (5, 6))))
        # Test 1-d input
        vector = UncertaintyBase.check_2d_inputs(params, array((1, 2)))
        self.assertTrue(allclose(vector, array(([1], [2]))))
        # Test 1-row 2-d input
        vector = UncertaintyBase.check_2d_inputs(params, array(((1, 2, 3),
            (1, 2, 3))))
        self.assertTrue(allclose(vector, array(((1, 2, 3), (1, 2, 3)))))

    @SkipTest
    def test_check_bounds_reasonableness(self):
        params = self.make_params_array(1)
        params['maximum'] = -0.3
        params['loc'] = 1
        params['scale'] = 1
        self.assertRaises(UnreasonableBoundsError,
            NormalUncertainty.check_bounds_reasonableness, params)

    def test_bounded_random_variables(self):
        params = self.make_params_array(1)
        params['maximum'] = -0.2 # Only ~ 10 percent of distribution
        params['loc'] = 1
        params['scale'] = 1
        sample = NormalUncertainty.bounded_random_variables(params, size=50000,
            maximum_iterations=1000)
        self.assertEqual((sample > -0.2).sum(), 0)
        self.assertEqual(sample.shape, (1, 50000))
        self.assertTrue(abs(sample.sum()) > 0)

    def test_bounded_uncertainty_base_validate(self):
        """BoundedUncertaintyBase: Make sure legitimate bounds are provided"""
        params = self.make_params_array(1)
        # Only maximum
        params['maximum'] = 1
        params['minimum'] = NaN
        self.assertRaises(ImproperBoundsError, BoundedUncertaintyBase.validate,
            params)
        # Only minimum
        params['maximum'] = NaN
        params['minimum'] = -1
        self.assertRaises(ImproperBoundsError, BoundedUncertaintyBase.validate,
            params)

    def test_undefined_uncertainty(self):
        params = self.make_params_array(1)
        self.assertRaises(UndefinedDistributionError, UndefinedUncertainty.cdf,
            params, random.random(10))
        params = self.make_params_array(2)
        params['loc'] = 9
        self.assertTrue(allclose(ones((2,3))*9,
            UndefinedUncertainty.random_variables(params, 3)))
        random_percentages = random.random(20).reshape(2, 10)
        self.assertTrue(allclose(ones((2,10))*9,
            UndefinedUncertainty.ppf(params, random_percentages)))


class LognormalTestCase(UncertaintyTestCase):
    def pdf(self, x, mu, scale):
        return 1/(x*sqrt(2*pi*scale**2))*e**(-((log(x)-mu)**2)/(
            2*scale**2))

    def cdf(self, x, mu, scale):
        return 0.5 + 0.5*erf((log(x) - mu)/sqrt(2*pi*scale**2))

    def test_lognormal_implementation(self):
        """Test lognormal implementation against known and calculated values.

Every time I re-learn how SciPy does lognormal I get confused again. Imagine a lognormal distribution with a geometric mean of 0.5, an underlying mean of log(0.5) (this is the mean of the underlying normal distribution), and a scale (standard deviation) of 0.5.

The formula for PDF is::

    1/(x*sqrt(2*pi*scale**2))*e**(-((log(x)-mu)**2)/(2*scale**2))

The formula for CDF is::

    0.5 + 0.5*erf((log(x) - mu)/sqrt(2*pi*scale**2))

Known values: The geometric mean is also the median.

* CDF(x=0.5, mu=log(0.5), scale=1) = 0.5
* PPF(x=0.5, mu=log(0.5), scale=1) = 0.5
* PDF(x=1, mu=0, scale=1) = 1/sqrt(2*pi) = 0.398942280401
* PDF(x=0.8, mu=log(0.7), scale=0.6) = 1/(0.8*sqrt(2*pi*0.36))*e**(-((log(0.8/0.7))**2)/0.72) = 0.81079978795919017

        """
        # Test manual definitions
        self.assertTrue(allclose(self.cdf(0.5, log(0.5), 1), 0.5))
        self.assertTrue(allclose(self.pdf(1, 0, 1), 1/sqrt(2*pi)))
        # Test known values
        params = self.make_params_array()
        params['loc'] = 0.5
        params['scale'] = 1
        self.assertTrue(allclose(LognormalUncertainty.ppf(params.copy(),
            array([[0.5]]), transform=True), 0.5))
        self.assertTrue(allclose(LognormalUncertainty.cdf(params.copy(),
            array([[0.5]]), transform=True), 0.5))
        params['loc'] = log(0.5)
        self.assertTrue(allclose(LognormalUncertainty.ppf(params.copy(),
            array([[0.5]])), 0.5))
        self.assertTrue(allclose(LognormalUncertainty.cdf(params.copy(),
            array([[0.5]])), 0.5))
        # Similarly for small means
        params = self.make_params_array()
        params['loc'] = 0.05
        params['scale'] = 1
        self.assertTrue(allclose(LognormalUncertainty.ppf(params.copy(),
            array([[0.5]]), transform=True), 0.05))
        self.assertTrue(allclose(LognormalUncertainty.cdf(params.copy(),
            array([[0.05]]), transform=True), 0.5))
        params['loc'] = log(0.05)
        self.assertTrue(allclose(LognormalUncertainty.ppf(params.copy(),
            array([[0.5]])), 0.05))
        self.assertTrue(allclose(LognormalUncertainty.cdf(params.copy(),
            array([[0.05]])), 0.5))
        params['loc'] = 1
        self.assertTrue(allclose(LognormalUncertainty.pdf(params.copy(),
            array([[1]]), transform=True)[1], 1/sqrt(2*pi)))
        params['loc'] = 0
        self.assertTrue(allclose(LognormalUncertainty.pdf(params.copy(),
            array([[1]]))[1], 1/sqrt(2*pi)))
        params['loc'] = 0.7
        params['scale'] = 0.6
        self.assertTrue(allclose(LognormalUncertainty.pdf(params.copy(),
            array([[0.8]]), transform=True)[1], 0.81079978795919017))

    def test_lognormal_transform_negative(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        params[1]['minimum'] = -4
        params[1]['loc'] = -3
        params[1]['maximum'] = -1
        params['scale'] = 0.8
        oneDtransformed = oneDparams.copy()
        LognormalUncertainty.transform_negative(oneDtransformed)
        self.assertTrue(allclose(oneDtransformed['loc'], log(array((3,)))))
        self.assertFalse(oneDtransformed['negative'])
        paramstransformed = params.copy()
        LognormalUncertainty.transform_negative(paramstransformed)
        self.assertTrue(allclose(paramstransformed['negative'], array((False,
            True))))
        self.assertTrue(allclose(paramstransformed['loc'],
            log(array((3,3)))))

    def test_lognormal_validate_required_values(self):
        params = self.make_params_array(1)
        params['loc'] = 1
        self.assertRaises(InvalidParamsError, LognormalUncertainty.validate,
            params, transform=True)
        params['scale'] = 0.8
        params['loc'] = NaN
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
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(allclose(array([[0.83281874,3.00000006,6.96037632]]),
            LognormalUncertainty.ppf(oneDparams, array([[0.1, 0.5, 0.8]]),
            transform=True)))
        self.assertTrue(allclose(array([[0.83281874], [3.00000006]]),
            LognormalUncertainty.ppf(params, array([0.1, 0.5]), transform=True)))

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
        self.assertTrue(allclose(array([[-0.83281874,-3.00000006,-6.96037632]]),
            LognormalUncertainty.ppf(oneDparams, array([[0.1, 0.5, 0.8]]),
            transform=True)))
        self.assertTrue(allclose(array([[-0.83281874], [-3.00000006]]),
            LognormalUncertainty.ppf(params, array([0.1, 0.5]),
            transform=True)))

    def test_lognormal_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(allclose(array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(oneDparams,
            array([[1,2,3,4]]), transform=True)))
        self.assertTrue(allclose(array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(params, array([[1,2,3,4],
            [1,2,3,4]]), transform=True)[1,:]))

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
        self.assertTrue(allclose(array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(oneDparams.copy(),
            -1*array([[1,2,3,4]]), transform=True)))
        self.assertTrue(allclose(array([[0.1359686,0.34256782,0.49999999,
            0.61320494]]), LognormalUncertainty.cdf(params.copy(),
            -1*array([[1,2,3,4], [1,2,3,4]]), transform=True)[1,:]))

    def test_lognormal_seeded_random(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertTrue(allclose(array([[0.66761271]]),
            LognormalUncertainty.random_variables(oneDparams, 1,
            self.seeded_random(), transform=True)))

    def test_lognormal_negative_seeded_random(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        self.assertTrue(allclose(array([[-0.66761271]]),
            LognormalUncertainty.random_variables(oneDparams, 1,
            self.seeded_random(), transform=True)))

    def test_lognormal_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        variables = LognormalUncertainty.random_variables(oneDparams,
            size=50000, transform=True)
        self.assertEqual((1, 50000), variables.shape)
        self.assertTrue(4 < average(variables) < 4.25)
        self.assertTrue(2.9 < median(variables) < 3.1)
        variables = LognormalUncertainty.random_variables(params, size=50000,
            transform=True)
        self.assertEqual((2,50000), variables.shape)
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertTrue(4 < average(variables[0,:]) < 4.25)
        self.assertTrue(2.9 < median(variables[1,:]) < 3.1)
        self.assertTrue(4 < average(variables[1,:]) < 4.25)

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
        self.assertTrue(4 < -1 * average(variables) < 4.25)
        self.assertTrue(2.9 < -1 * median(variables) < 3.1)
        variables = LognormalUncertainty.random_variables(params, size=50000,
            transform=True)
        self.assertEqual((2,50000), variables.shape)
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertTrue(4 < -1 * average(variables[0,:]) < 4.25)
        self.assertTrue(2.9 < -1 * median(variables[1,:]) < 3.1)
        self.assertTrue(4 < -1 * average(variables[1,:]) < 4.25)

    def test_lognormal_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertEqual({'upper': 14.859097922170479,
            'lower': 0.60568955155650162, 'median': 3.0000000595022631,
            'mode': 1.5818772733323232, 'mean': 4.131383414350033},
            LognormalUncertainty.statistics(oneDparams, transform=True))

    def test_lognormal_negative_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        self.assertEqual({'lower': -14.859097922170479,
            'upper': -0.60568955155650162, 'median': -3.0000000595022631,
            'mode': -1.5818772733323232, 'mean': -4.131383414350033},
            LognormalUncertainty.statistics(oneDparams, transform=True))

    def test_lognormal_bounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        points = array([[1,2,3,4]])
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), points,
            transform=True)
        self.assertTrue(allclose(points, xs))
        self.assertTrue(allclose(ys, array([self.pdf(x, log(3), 0.8) for x \
            in points])))
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), transform=True)
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertEqual(xs.min(), 1)
        self.assertTrue(3.97 < xs.max() <= 4)
        stats = LognormalUncertainty.statistics(oneDparams, transform=True)
        self.assertTrue(abs(ys.min() - min(self.pdf(1, log(3), 0.8),
            self.pdf(4, log(3), 0.8))) / ys.min() < 0.01)
        self.assertTrue(abs(ys.max() - self.pdf(stats['mode'], log(3), 0.8)
            ) / ys.max() < 0.01)
        self.assertTrue(allclose(ys[-1], self.pdf(xs[-1], log(3), 0.8)))
        self.assertTrue(allclose(ys[0], self.pdf(xs[0], log(3), 0.8)))

    def test_lognormal_negative_bounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = -4
        oneDparams['loc'] = -3
        oneDparams['maximum'] = -1
        points = -1 * array([[1,2,3,4]])
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), points,
            transform=True)
        self.assertTrue(allclose(points, xs))
        self.assertTrue(allclose(ys, array([self.pdf(x, log(3), 0.8) for x \
            in -1*points])))
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), transform=True)
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertTrue(-4 <= xs.min() <= -3.98)
        self.assertEqual(xs.max(), -1)
        self.assertTrue(abs(ys.min() - min(self.pdf(1, log(3), 0.8),
            self.pdf(4, log(3), 0.8))) / ys.min() < 0.01)
        stats = LognormalUncertainty.statistics(oneDparams, transform=True)
        self.assertTrue(abs(ys.max() - self.pdf(abs(stats['mode']), log(3),
            0.8)) / ys.max() < 0.01)
        self.assertTrue(allclose(ys[-1], self.pdf(abs(xs[-1]), log(3), 0.8)))
        self.assertTrue(allclose(ys[0], self.pdf(abs(xs[0]), log(3), 0.8)))

    def test_lognormal_unbounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = oneDparams['maximum'] = NaN
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), transform=True)
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        min_x = 1e-9
        max_x = exp(log(3)+0.8 * \
            LognormalUncertainty.standard_deviations_in_default_range)
        self.assertTrue(abs(xs.max() - max_x) / xs.max() < 0.01)
        self.assertTrue(allclose(xs.min(), min_x))
        stats = LognormalUncertainty.statistics(oneDparams, transform=True)
        self.assertTrue(abs(ys.max() - self.pdf(stats['mode'], log(3),
            0.8)) / ys.max() < 0.01)
        self.assertTrue(abs(ys.min() - min(self.pdf(min_x, log(3), 0.8),
            self.pdf(max_x, log(3), 0.8))) / ys.min() < 0.01)

    def test_lognormal_negative_unbounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = oneDparams['maximum'] = NaN
        oneDparams['loc'] = -3
        xs, ys = LognormalUncertainty.pdf(oneDparams.copy(), transform=True)
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        min_x = -exp(log(3)+0.8 * \
            LognormalUncertainty.standard_deviations_in_default_range)
        max_x = -1e-9
        self.assertTrue(abs(xs.min() - min_x) / xs.min() < 0.01)
        self.assertTrue(allclose(xs.max(), max_x))
        stats = LognormalUncertainty.statistics(oneDparams, transform=True)
        self.assertTrue(abs(ys.max() - self.pdf(abs(stats['mode']), log(3),
            0.8)) / ys.max() < 0.01)
        self.assertTrue(abs(ys.min() - min(self.pdf(abs(min_x), log(3), 0.8),
            self.pdf(abs(max_x), log(3), 0.8))) / ys.min() < 0.01)


class NormalTestCase(UncertaintyTestCase):
    def test_normal_validate(self):
        params = self.make_params_array(1)
        params['scale'] = NaN
        self.assertRaises(InvalidParamsError, NormalUncertainty.validate,
            params)

    def test_normal_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(allclose(array([[1.97475873, 3, 3.673297]]),
            NormalUncertainty.ppf(oneDparams, array([[0.1, 0.5, 0.8]]))))
        self.assertTrue(allclose(array([[1.97475873], [3]]),
            NormalUncertainty.ppf(params, array([0.1, 0.5]))))

    def test_normal_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        self.assertTrue(allclose(array([[0.00620967, 0.10564978, 0.5,
            0.89435022]]), NormalUncertainty.cdf(oneDparams,
            array([[1,2,3,4]]))))
        self.assertTrue(allclose(array([[0.00620967, 0.10564978, 0.5,
            0.89435022]]), NormalUncertainty.cdf(params, array([[1,2,3,4],
            [1,2,3,4]]))[1,:]))

    def test_normal_seeded_random(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertTrue(allclose(array([[ 1.49734064]]),
            NormalUncertainty.random_variables(oneDparams, 1,
            self.seeded_random())))

    def test_normal_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        oneDparams['scale'] = 0.8
        params['scale'] = 0.8
        variables = NormalUncertainty.random_variables(oneDparams, size=50000)
        self.assertEqual((1, 50000), variables.shape)
        self.assertTrue(2.95 < average(variables) < 3.05)
        self.assertTrue(2.95 < median(variables) < 3.05)
        variables = NormalUncertainty.random_variables(params, size=50000)
        self.assertEqual((2,50000), variables.shape)
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertTrue(2.95 < average(variables[1,:]) < 3.05)
        self.assertTrue(2.95 < median(variables[1,:]) < 3.05)
        self.assertTrue(2.95 < average(variables[0,:]) < 3.05)

    def test_normal_statistics(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        self.assertEqual({'upper': 4.5999999046325684,
            'lower': 1.3999999761581421, 'median': 3.0, 'mode': 3.0,
            'mean': 3.0}, NormalUncertainty.statistics(oneDparams))

    def test_normal_bounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        points = array([[1,2,3,4]])
        xs, ys = NormalUncertainty.pdf(oneDparams.copy(), points)
        self.assertTrue(allclose(points, xs))
        self.assertTrue(allclose(array([0.02191038, 0.22831136, 0.49867784,
            0.22831136]), ys))
        xs, ys = NormalUncertainty.pdf(oneDparams.copy())
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertEqual(xs.min(), 1)
        self.assertTrue(3.98 < xs.max() <= 4)
        self.assertEqual(ys.min(), 0.021910377331033407)
        self.assertTrue(allclose(ys.max(), 0.498668095951))

    def test_normal_unbounded_pdf(self):
        oneDparams = self.biased_params_1d()
        oneDparams['scale'] = 0.8
        oneDparams['minimum'] = oneDparams['maximum'] = NaN
        xs, ys = NormalUncertainty.pdf(oneDparams.copy())
        self.assertEqual(xs.shape, (200,))
        self.assertEqual(ys.shape, (200,))
        self.assertTrue(allclose(xs.min(), 1.23999989033))
        self.assertTrue(allclose(xs.max(), 4.74241173267))
        self.assertTrue(allclose(average(xs), 2.9912058115))
        self.assertTrue(allclose(ys.min(), 0.0443432302212))
        self.assertTrue(allclose(ys.max(), 0.498677843058))
        self.assertTrue(allclose(average(ys), 0.276188653511))


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
        self.assertTrue(allclose(array([1,2,3]),
            UniformUncertainty.ppf(oneDparams, array([[0, 0.5, 1]]))))
        self.assertTrue(allclose(array([[1],[2]]),
            UniformUncertainty.ppf(params, array([0, 0.5]))))

    def test_uniform_cdf(self):
        oneDparams = self.unif_params_1d()
        params = self.unif_params_2d()
        self.assertTrue(allclose(array([0, 0.5, 1]),
            UniformUncertainty.cdf(oneDparams, array([[1,2,3,]]))))
        self.assertTrue(allclose(array([[0], [0.5]]),
            UniformUncertainty.cdf(params, array([1,2]))))

    def test_uniform_seeded_random(self):
        oneDparams = self.unif_params_1d()
        self.assertTrue(allclose(2.15281272,
            UniformUncertainty.random_variables(oneDparams, 1,
            self.seeded_random())))

    def test_uniform_random(self):
        oneDparams = self.unif_params_1d()
        params = self.unif_params_2d()
        variables = UniformUncertainty.random_variables(oneDparams, size=5000)
        self.assertEqual((1, 5000), variables.shape)
        self.assertTrue(1.95 < average(variables) < 2.05)
        variables = UniformUncertainty.random_variables(params, size=5000)
        self.assertEqual((2,5000), variables.shape)
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertTrue(1.95 < average(variables[0,:]) < 2.05)
        self.assertTrue(1.95 < average(variables[1,:]) < 2.05)

    def test_uniform_statistics(self):
        oneDparams = self.unif_params_1d()
        self.assertEqual({'mean': 2, 'mode': 2, 'median': 2, 'lower': 1,
            'upper': 3}, UniformUncertainty.statistics(oneDparams))

    def test_uniform_pdf(self):
        oneDparams = self.unif_params_1d()
        xs, ys = UniformUncertainty.pdf(oneDparams)
        self.assertTrue(allclose(array([1, 3]), xs))
        self.assertTrue(allclose(array([0.5, 0.5]), ys))
        points = array([1, 2, 3])
        xs, ys = UniformUncertainty.pdf(oneDparams, points)
        self.assertTrue(allclose(points, xs))
        self.assertTrue(allclose(array([0.5, 0.5, 0.5]), ys))


class TriangularTestCase(UncertaintyTestCase):
    def test_triangular(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()

    def test_triangular_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        percentages = array([[0, 0.5, 1]])
        self.assertTrue(allclose(TriangularUncertainty.ppf(oneDparams,
            percentages), array([1, 2.73205083, 4])))
        self.assertTrue(allclose(TriangularUncertainty.ppf(params,
            array([[0, 0.5, 1], [0, 0.5, 1]])), array([[ 1, 2.73205083, 4], [ 1,
            2.73205083, 4]])))

    def test_triangular_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        percentages = array([0, 0.5, 1])
        self.assertTrue(allclose(TriangularUncertainty.cdf(oneDparams,
            array([[1, 2.73205083, 4]])), percentages))
        self.assertTrue(allclose(TriangularUncertainty.cdf(params,
            array([1, 2.73205083])), array([[0], [0.5]])))

    def test_triangular_seeded_random(self):
        oneDparams = self.biased_params_1d()
        self.assertTrue(allclose(2.85968765,
            TriangularUncertainty.random_variables(oneDparams, 1,
            self.seeded_random())))

    def test_triangular_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        variables = TriangularUncertainty.random_variables(oneDparams, 5000)
        self.assertTrue(2.61 < average(variables) < 2.71)
        self.assertEqual((1, 5000), variables.shape)
        variables = TriangularUncertainty.random_variables(params, 5000)
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertFalse(allclose(variables[0,:], variables[1,:]))
        self.assertTrue(2.61 < average(variables[0,:]) < 2.71)
        self.assertTrue(2.61 < average(variables[1,:]) < 2.71)

    def test_triangular_statistics(self):
        oneDparams = self.biased_params_1d()
        tri_stats = {'upper': None, 'lower': None, 'median': 2.732050895690918,
            'mode': 3.0, 'mean': 2.6666667461395264}
        self.assertEqual(TriangularUncertainty.statistics(oneDparams),
            tri_stats)

    def test_triangular_pdf(self):
        oneDparams = self.biased_params_1d()
        xs, ys = TriangularUncertainty.pdf(oneDparams)
        self.assertTrue(allclose(array([1, 3, 4]), xs))
        self.assertTrue(allclose(array([0, 0.66666669, 0]), ys))
        points = array([1, 2, 3, 4])
        xs, ys = TriangularUncertainty.pdf(oneDparams, points)
        self.assertTrue(allclose(points, xs))
        self.assertTrue(allclose(array([0, 0.99999997, 1.99999994, 0]), ys))


class BernoulliTestCase(UncertaintyTestCase):
    def test_bernoulli_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        self.assertTrue(allclose(BernoulliUncertainty.ppf(oneDparams,
            array([[0, 0.25, 0.5, 0.75, 1]])), array([[1,1,1,4,4]])))
        self.assertTrue(allclose(BernoulliUncertainty.ppf(params,
            array([0.5, 0.8])), array([[1],[4]])))

    def test_bernoulli_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        self.assertTrue(allclose(BernoulliUncertainty.cdf(oneDparams,
            array([[1,2,3,4]])), array([[0,0,1,1]])))
        self.assertTrue(allclose(BernoulliUncertainty.cdf(params,
            array([1,3])), array([[0],[1]])))

    def test_bernoulli_seeded_random(self):
        oneDparams = self.biased_params_1d()
        self.assertTrue(allclose(array([[0,1,0,1,1,0,0,0,0,0]]),
            BernoulliUncertainty.random_variables(oneDparams, 10,
            self.seeded_random())))

    def test_bernoulli_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        variables = BernoulliUncertainty.random_variables(oneDparams, 50000)
        self.assertTrue(0.3 < average(variables) < 0.35)
        self.assertEqual(variables.shape, (1, 50000))
        variables = BernoulliUncertainty.random_variables(params, 50000)
        self.assertTrue(0.3 < average(variables[0,:]) < 0.35)
        self.assertTrue(0.3 < average(variables[1,:]) < 0.35)
        self.assertEqual(variables.shape, (2, 50000))
        self.assertFalse(allclose(variables[0,:], variables[1,:]))

    def test_bernoulli_statistics(self):
        oneDparams = self.biased_params_1d()
        bern_stats = {'upper': None, 'lower': None, 'median': None,
            'mode': None, 'mean': 3}
        self.assertEqual(BernoulliUncertainty.statistics(oneDparams),
            bern_stats)

    def test_bernoulli_pdf(self):
        oneDparams = self.biased_params_1d()
        self.assertRaises(NotImplementedError, BernoulliUncertainty.pdf,
            oneDparams)

