from __future__ import division
from ...distributions import StudentsTUncertainty
from ...errors import InvalidParamsError
from ..base import UncertaintyTestCase
import numpy as np


class StudentsTTestCase(UncertaintyTestCase):

    def pretty_close(self, a, b):
        if b == 0:
            self.assertTrue(a - 0.1 < b < a + 0.1)
        else:
            self.assertTrue(0.95 * a < b < 1.05 * a)

    def test_loc_and_scale_nan(self):
        params = self.make_params_array()
        params['shape'] = 1
        sample = StudentsTUncertainty.random_variables(params, 5000)
        self.pretty_close(np.median(sample), 0)

    def test_loc_matters(self):
        params = self.make_params_array()
        params['shape'] = 1
        params['loc'] = 10
        sample = StudentsTUncertainty.random_variables(params, 1000)
        self.pretty_close(np.median(sample), 10)

    def test_scale_matters(self):
        params = self.make_params_array()
        params['shape'] = 1
        sample_1 = StudentsTUncertainty.random_variables(params, 5000)
        params['scale'] = 5
        sample_2 = StudentsTUncertainty.random_variables(params, 5000)
        self.assertTrue(np.std(sample_1) < np.std(sample_2))

    def test_random_variables(self):
        params = self.make_params_array()
        params['shape'] = 5
        sample = StudentsTUncertainty.random_variables(params, 20000)
        # nu / (nu - 2) if nu > 2
        expected_variance = 5. / 3
        self.pretty_close(np.var(sample), expected_variance)

    def test_scale_validation(self):
        params = self.make_params_array()
        params['shape'] = 1
        # NaN is OK
        StudentsTUncertainty.validate(params)
        # > 0 is OK
        params['scale'] = 1
        StudentsTUncertainty.validate(params)
        # <= 0 is not
        params['scale'] = 0
        self.assertRaises(
            InvalidParamsError,
            StudentsTUncertainty.validate,
            params
        )
        params['scale'] = -1
        self.assertRaises(
            InvalidParamsError,
            StudentsTUncertainty.validate,
            params
        )

    def test_shape_validation(self):
        params = self.make_params_array()
        params['shape'] = 1
        # > 0 is OK
        params['shape'] = 1
        StudentsTUncertainty.validate(params)
        # <= 0 is not
        params['shape'] = 0
        self.assertRaises(
            InvalidParamsError,
            StudentsTUncertainty.validate,
            params
        )
        params['shape'] = -1
        self.assertRaises(
            InvalidParamsError,
            StudentsTUncertainty.validate,
            params
        )
