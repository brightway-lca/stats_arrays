import unittest
from numpy import *
from ..random import RandomNumberGenerator
from ..distributions import *
from ..errors import ImproperBoundsError, UnknownUncertaintyType
import numpy as np


class RandomNumberGeneratorTestCase(unittest.TestCase):
    def make_params_array(self, length=1):
        assert isinstance(length, int)
        params = zeros((length,), dtype=[
            ('input', 'u4'),
            ('output', 'u4'),
            ('loc', 'f4'),
            ('negative', 'b1'),
            ('scale', 'f4'),
            ('minimum', 'f4'),
            ('maximum', 'f4')
        ])
        params['minimum'] = params['maximum'] = params['scale'] = np.NaN
        return params

    def test_improper_bounds(self):
        params_array = self.make_params_array()
        params_array[0] = (0, 0, 0, False, 0.2, 1, 0)
        self.assertRaises(ImproperBoundsError, RandomNumberGenerator,
            LognormalUncertainty, params_array)
        params_array[0] = (0, 0, 0, False, 0.2, 1, 0)
        self.assertRaises(ImproperBoundsError, RandomNumberGenerator,
            TriangularUncertainty, params_array)
        params_array[0] = (0, 0, 2, False, 0.2, 0, 1)
        self.assertRaises(ImproperBoundsError, RandomNumberGenerator,
            TriangularUncertainty, params_array)
        params_array[0] = (0, 0, -1, False, 0.2, 0, 1)
        self.assertRaises(ImproperBoundsError, RandomNumberGenerator,
            UniformUncertainty, params_array)
        # assertTrue is a synonym for no exception raised
        params_array[0] = (0, 0, 1, False, 0.2, 0, 1)
        self.assertTrue(RandomNumberGenerator(TriangularUncertainty,
            params_array))
        params_array[0] = (0, 0, 0, False, 0.2, 0, 1)
        self.assertTrue(RandomNumberGenerator(TriangularUncertainty,
            params_array))

    def test_invalid_uncertainty_type(self):
        params_array = self.make_params_array()
        params_array[0] = (0, 0, 0.5, False, 0.2, 0, 1)
        self.assertRaises(UnknownUncertaintyType, RandomNumberGenerator, 1,
            params_array)

    def test_unbounded_lognormal(self):
        params_array = self.make_params_array()
        params_array[0] = (0, 0, 1, False, 0.2, np.NaN, np.NaN)
        data = RandomNumberGenerator(LognormalUncertainty,
            params_array, size=100).next()
        self.assertTrue(0.8 < median(data) < 1.2)

    def test_negative_lognormal(self):
        params_array = self.make_params_array()
        params_array[0] = (0, 0, -1, False, 0.2, np.NaN, np.NaN)
        R = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100,
            convert_lognormal=True
        )
        data = R.next()
        self.assertTrue(-0.8 > median(data) > -1.2)

        params_array[0] = (0, 0, -1, True, 0.2, np.NaN, np.NaN)
        R = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100,
            convert_lognormal=True
        )
        data = R.next()
        self.assertTrue(-0.8 > median(data) > -1.2)

        params_array[0] = (0, 0, 1, True, 0.2, np.NaN, np.NaN)
        R = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100
        )
        data = R.next()
        self.assertTrue(-0.8 > median(data) > -1.2)

    def test_bounded_lognormal(self):
        params_array = self.make_params_array()
        params_array[0] = (0, 0, 1, False, 0.2, 0.8, np.NaN)
        data = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100
        ).next()
        self.assertTrue(0.8 < np.median(data) < 1.2)
        self.assertTrue(data.min() >= 0.8)

        params_array[0] = (0, 0, 1, False, 0.2, np.NaN, 1.2)
        data = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100
        ).next()
        self.assertTrue(0.8 < median(data) < 1.2)
        self.assertTrue(data.max() <= 1.2)

        params_array[0] = (0, 0, 1, False, 0.2, 0.8, 1)
        data = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100
        ).next()
        self.assertTrue(0.8 < median(data) < 1)
        self.assertTrue(data.min() >= 0.8)
        self.assertTrue(data.max() <= 1)

        params_array[0] = (0, 0, -1, False, 0.2, -1.1, -0.5)
        data = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100,
            convert_lognormal=True
        ).next()
        self.assertTrue(-1.1 < median(data) < -0.5)
        self.assertTrue(data.min() >= -1.1)
        self.assertTrue(data.max() <= -0.5)

    def test_bound_interval_multiple_params(self):
        params_array = self.make_params_array(2)
        params_array[0] = (0, 0, 1, False, 0.2, 0.8, 1)
        params_array[1] = (0, 0, 1, False, 0.2, 0.8, 1)
        data = RandomNumberGenerator(
            LognormalUncertainty,
            params_array,
            size=100
        ).next()
        self.assertTrue(0.8 < median(data) < 1)
        self.assertTrue(data.min() >= 0.8)
        self.assertTrue(data.max() <= 1)
        self.assertEqual(data.shape, (2, 100))

    def test_no_uncertainty(self):
        params_array = self.make_params_array(2)
        params_array[0] = (0,0,1, False, np.NaN, np.NaN, np.NaN)
        params_array[1] = (0,0,1, False, np.NaN, np.NaN, np.NaN)
        data = RandomNumberGenerator(NoUncertainty, params_array,
            size=10).next()
        self.assertTrue(allclose(data, ones((2,10))))
        data = RandomNumberGenerator(UndefinedUncertainty,
            params_array, size=10).next()
        self.assertTrue(allclose(data, ones((2,10))))

    def test_data_size(self):
        params_array = self.make_params_array(2)
        params_array[0] = (0,0,1, True, 0.5, 0, 2)
        params_array[1] = (0,0,1, True, 0.5, 0, 2)
        data = RandomNumberGenerator(NoUncertainty, params_array,
            size=10).next()
        self.assertEqual(data.shape, (2, 10))
        data = RandomNumberGenerator(NormalUncertainty, params_array,
            size=10, convert_lognormal=True).next()
        self.assertEqual(data.shape, (2, 10))
        params_array = self.make_params_array()
        params_array[0] = (0,0,1, False, 0.2, 0.8, np.NaN)
        data = RandomNumberGenerator(NoUncertainty, params_array,
            size=10).next()
        self.assertEqual(data.shape, (1, 10))
        data = RandomNumberGenerator(NormalUncertainty, params_array,
            size=10).next()
        self.assertEqual(data.shape, (1, 10))

    def test_bernoulli(self):
        params_array = self.make_params_array(2)
        params_array[0] = (0, 0, 0.2, True, np.NaN, 0, 1)
        params_array[1] = (0, 0, 0.8, True, np.NaN, 0, 1)
        data = RandomNumberGenerator(BernoulliUncertainty,
            params_array, size=500).next()
        self.assertTrue(0.45 < average(data) < 0.55)
        self.assertTrue(0.15 < average(data[1, :]) < 0.25)
        self.assertTrue(0.75 < average(data[0, :]) < 0.85)

    def test_seed(self):
        params_array = self.make_params_array()
        params_array[0] = (0,0,1, False, 0.2, 0.8, np.NaN)
        R = RandomNumberGenerator(LognormalUncertainty, params_array.copy(),
            size=1, convert_lognormal=True, seed=111)
        self.assertTrue(allclose(array([ 1.07989503]), R.next()[0]))

