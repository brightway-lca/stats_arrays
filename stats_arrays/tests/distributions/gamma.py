from __future__ import division
from ...distributions import GammaUncertainty
from ...errors import InvalidParamsError
from ..base import UncertaintyTestCase
import numpy as np


class GammaUncertaintyTestCase(UncertaintyTestCase):

    def pretty_close(self, a, b):
        if b == 0:
            self.assertTrue(a - 0.05 < b < a + 0.05)
        else:
            self.assertTrue(0.95 * a < b < 1.05 * a)

    def test_random_variables(self):
        params = self.make_params_array()
        params['shape'] = 2
        params['scale'] = 5
        sample = GammaUncertainty.random_variables(params, 10000)
        # Mean: shape * scale
        self.pretty_close(2 * 5, np.mean(sample))
        # Mean: shape * scale^2
        self.pretty_close(2 * 5 ** 2, np.var(sample))

    def test_random_variables_2d(self):
        params = self.make_params_array(2)
        params['shape'] = (2, 3)
        params['scale'] = (5, 10)
        sample = GammaUncertainty.random_variables(params, 10000)
        self.pretty_close(2 * 5, np.mean(sample[0, :]))
        self.pretty_close(3 * 10, np.mean(sample[1, :]))

    def test_random_variables_offset(self):
        params = self.make_params_array(2)
        params['shape'] = (2, 3)
        params['scale'] = (5, 10)
        params['loc'] = (100, np.NaN)
        sample = GammaUncertainty.random_variables(params, 10000)
        self.pretty_close(2 * 5 + 100, np.mean(sample[0, :]))
        self.pretty_close(3 * 10, np.mean(sample[1, :]))

    def test_loc_nan_ok(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = 1
        params['shape'] = 1
        GammaUncertainty.validate(params)
        return True

    def test_scale_validation(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = -1
        params['shape'] = 1
        self.assertRaises(
            InvalidParamsError,
            GammaUncertainty.validate,
            params
        )

    def test_shape_validation(self):
        params = self.make_params_array()
        params['loc'] = np.NaN
        params['scale'] = 1
        params['shape'] = -1
        self.assertRaises(
            InvalidParamsError,
            GammaUncertainty.validate,
            params
        )
