from __future__ import division
from ...distributions import WeibullUncertainty
from ...errors import InvalidParamsError
from ..base import UncertaintyTestCase
import numpy as np


class WeibullTestCase(UncertaintyTestCase):

    def pretty_close(self, a, b):
        if b == 0:
            self.assertTrue(a - 0.05 < b < a + 0.05)
        else:
            self.assertTrue(0.95 * a < b < 1.05 * a)

    def test_random_variables(self):
        params = self.make_params_array()
        params['scale'] = 2  # lambda
        params['shape'] = 5  # k
        sample = WeibullUncertainty.random_variables(params, 10000)
        # Median: lambda * ln(2)^(1/k)
        self.pretty_close(2 * np.log(2) ** (1 / 5), np.median(sample))

    def test_random_variables_2d(self):
        params = self.make_params_array(2)
        params['scale'] = (5, 10)
        params['shape'] = (2, 3)
        sample = WeibullUncertainty.random_variables(params, 10000)
        self.pretty_close(5 * np.log(2) ** (1 / 2), np.median(sample[0, :]))
        self.pretty_close(10 * np.log(2) ** (1 / 3), np.median(sample[1, :]))

    def test_random_variables_offset(self):
        params = self.make_params_array(2)
        params['scale'] = (5, 10)
        params['shape'] = (2, 3)
        params['loc'] = (100, np.NaN)
        sample = WeibullUncertainty.random_variables(params, 10000)
        self.pretty_close(100 + 5 * np.log(2) ** (1 / 2), np.median(sample[0, :]))
        self.pretty_close(10 * np.log(2) ** (1 / 3), np.median(sample[1, :]))

    def test_loc_nan_ok(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = 1
        params['shape'] = 1
        WeibullUncertainty.validate(params)
        return True

    def test_scale_validation(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = -1
        params['shape'] = 1
        self.assertRaises(
            InvalidParamsError,
            WeibullUncertainty.validate,
            params
        )

    def test_shape_validation(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = 1
        params['shape'] = -1
        self.assertRaises(
            InvalidParamsError,
            WeibullUncertainty.validate,
            params
        )
