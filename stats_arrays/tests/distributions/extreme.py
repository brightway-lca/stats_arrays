from __future__ import division
from ...distributions import GeneralizedExtremeValueUncertainty as GEVU
from ...errors import InvalidParamsError
from ..base import UncertaintyTestCase
import numpy as np


class GeneralizedExtremeValueUncertaintyTestCase(UncertaintyTestCase):

    def test_random_variables(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['scale'] = 5
        # Formula for median (loc - scale * ln ln 2)
        expected_median = 2 - 5 * np.log(np.log(2))
        results = GEVU.random_variables(params, 10000)
        found_median = np.median(results)
        self.assertEqual(results.shape, (1, 10000))
        self.assertTrue(0.95 * expected_median < found_median)
        self.assertTrue(found_median < 1.05 * expected_median)

    def test_loc_validation(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        self.assertRaises(
            InvalidParamsError,
            GEVU.validate,
            params
        )

    def test_scale_validation(self):
        params = self.make_params_array()
        params['scale'] = -1
        self.assertRaises(
            InvalidParamsError,
            GEVU.validate,
            params
        )

    def test_shape_validation(self):
        params = self.make_params_array()
        params['shape'] = 1
        self.assertRaises(
            InvalidParamsError,
            GEVU.validate,
            params
        )

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
        params['loc'] = params['scale'] = 1
        return params
