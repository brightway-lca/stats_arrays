import unittest
from ..distributions import *
from ..errors import ImproperBoundsError, \
    UndefinedDistributionError, InvalidParamsError, UnreasonableBoundsError
from nose.plugins.skip import SkipTest
import numpy as np


class UncertaintyTestCase(unittest.TestCase):

    def make_params_array(self, length=1):
        assert isinstance(length, int)
        params = np.zeros((length,), dtype=[
            ('input', 'u4'),
            ('output', 'u4'),
            ('loc', 'f4'),
            ('negative', 'b1'),
            ('scale', 'f4'),
            ('shape', 'f4'),
            ('minimum', 'f4'),
            ('maximum', 'f4')
        ])
        params['minimum'] = params['maximum'] = np.NaN
        params['scale'] = params['shape'] = np.NaN
        return params

    def seeded_random(self, seed=111111):
        return np.random.RandomState(seed)

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

    def right_triangles_min(self):
        params = self.make_params_array(1)
        params['minimum'] = 1
        params['loc'] = 1
        params['maximum'] = 4
        return params

    def right_triangles_max(self):
        params = self.make_params_array(1)
        params['minimum'] = 1
        params['loc'] = 4
        params['maximum'] = 4
        return params

class BaseTestCase(UncertaintyTestCase):

    def test_uncertainty_base_validate(self):
        """UncertaintyBase: Mean exists, and bounds are ok if present."""
        params = self.make_params_array(1)
        params['maximum'] = 2
        params['minimum'] = 2.1
        self.assertRaises(
            ImproperBoundsError,
            UncertaintyBase.validate,
            params
        )

    def test_check_2d_inputs(self):
        params = self.make_params_array(2)
        params['minimum'] = 0
        params['loc'] = 1
        params['maximum'] = 2
        # Params has 2 rows. The input vector can only have shape (2,) or (2, n)
        self.assertRaises(
            InvalidParamsError,
            UncertaintyBase.check_2d_inputs,
            params,
            np.array((1,))
        )
        self.assertRaises(
            InvalidParamsError,
            UncertaintyBase.check_2d_inputs,
            params,
            np.array(((1, 2),))
        )
        self.assertRaises(
            InvalidParamsError,
            UncertaintyBase.check_2d_inputs,
            params,
            np.array(((1, 2), (3, 4), (5, 6)))
        )
        # Test 1-d input
        vector = UncertaintyBase.check_2d_inputs(params, np.array((1, 2)))
        self.assertTrue(np.allclose(vector, np.array(([1], [2]))))
        # Test 1-row 2-d input
        vector = UncertaintyBase.check_2d_inputs(
            params,
            np.array(((1, 2, 3), (1, 2, 3)))
        )
        self.assertTrue(np.allclose(vector, np.array(((1, 2, 3), (1, 2, 3)))))

    @SkipTest
    def test_check_bounds_reasonableness(self):
        params = self.make_params_array(1)
        params['maximum'] = -0.3
        params['loc'] = 1
        params['scale'] = 1
        self.assertRaises(
            UnreasonableBoundsError,
            NormalUncertainty.check_bounds_reasonableness,
            params
        )

    def test_bounded_random_variables(self):
        params = self.make_params_array(1)
        params['maximum'] = -0.2  # Only ~ 10 percent of distribution
        params['loc'] = 1
        params['scale'] = 1
        sample = NormalUncertainty.bounded_random_variables(
            params,
            size=50000,
            maximum_iterations=1000
        )
        self.assertEqual((sample > -0.2).sum(), 0)
        self.assertEqual(sample.shape, (1, 50000))
        self.assertTrue(np.abs(sample.sum()) > 0)

    def test_bounded_uncertainty_base_validate(self):
        """BoundedUncertaintyBase: Make sure legitimate bounds are provided"""
        params = self.make_params_array(1)
        # Only maximum
        params['maximum'] = 1
        params['minimum'] = np.NaN
        self.assertRaises(
            ImproperBoundsError,
            BoundedUncertaintyBase.validate,
            params
        )
        # Only minimum
        params['maximum'] = np.NaN
        params['minimum'] = -1
        self.assertRaises(
            ImproperBoundsError,
            BoundedUncertaintyBase.validate,
            params
        )

    def test_undefined_uncertainty(self):
        params = self.make_params_array(1)
        self.assertRaises(
            UndefinedDistributionError,
            UndefinedUncertainty.cdf,
            params,
            np.random.random(10)
        )
        params = self.make_params_array(2)
        params['loc'] = 9
        self.assertTrue(np.allclose(
            np.ones((2, 3)) * 9,
            UndefinedUncertainty.random_variables(params, 3)
        ))
        random_percentages = np.random.random(20).reshape(2, 10)
        self.assertTrue(np.allclose(
            np.ones((2, 10)) * 9,
            UndefinedUncertainty.ppf(params, random_percentages)
        ))
