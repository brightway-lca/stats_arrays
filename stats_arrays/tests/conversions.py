from __future__ import print_function
import unittest
from ..distributions import UncertaintyBase
from ..utils import construct_params_array
import numpy as np


class ConversionTestCase(unittest.TestCase):

    def get_right_1d_array(self):
        params = construct_params_array(include_type=True)
        params['loc'] = 1
        params['maximum'] = 2
        params['minimum'] = 0
        params['scale'] = 3
        return params

    def get_right_2d_array(self):
        params = construct_params_array(2, include_type=True)
        params['loc'] = (1, 2)
        params['scale'] = (3, 4)
        params['maximum'] = 10
        params['minimum'] = 0
        return params

    def test_1d_dict(self):
        values = {"loc": 1.0,
                  "scale": 3.0,
                  "maximum": 2.0,
                  "minimum": 0.0,
                  "negative": False,
                  }
        self.sa_allclose(
            UncertaintyBase.from_dicts(values),
            self.get_right_1d_array(),
        )

    def test_1d_tuple(self):
        values = (1, 3, np.NaN, 0, 2, False, 0)
        self.sa_allclose(
            UncertaintyBase.from_tuples(values),
            self.get_right_1d_array(),
        )

    def test_2d_dict(self):
        values = (
            {"loc": 1.0,
             "scale": 3.0,
             "maximum": 10.0,
             "minimum": 0.0,
             "negative": False},
            {"loc": 2.0,
             "scale": 4.0,
             "maximum": 10.0,
             "minimum": 0.0,
             "negative": False}
        )
        self.sa_allclose(
            UncertaintyBase.from_dicts(*values),
            self.get_right_2d_array(),
        )

    def test_2d_tuple(self):
        values = (
            (1, 3, np.NaN, 0, 10, False, 0),
            (2, 4, np.NaN, 0, 10, False, 0),
        )
        self.sa_allclose(
            UncertaintyBase.from_tuples(*values),
            self.get_right_2d_array(),
        )

    def sa_allclose(self, a, b):
        """allclose for structured arrays"""
        for name in a.dtype.names:
            print(name, a[name], b[name])
            nan_mask_a = np.isnan(a[name])
            nan_mask_b = np.isnan(b[name])
            self.assertTrue(np.allclose(nan_mask_a, nan_mask_b))
            self.assertTrue(np.allclose(
                a[name][~nan_mask_a],
                b[name][~nan_mask_b]
            ))
        return True


def test_hetergeneous_from_dicts():
    dicts = [{
        'uncertainty type': 4,
        'minimum': 2,
        'maximum': 5,
        'amount': 3,
        'loc': 3,
    }, {
        'uncertainty_type': 2,
        'loc': 3,
        'scale': 0.2,
    }]
    answer = UncertaintyBase.from_dicts(*dicts)
    assert np.allclose(answer['uncertainty_type'], np.array((4, 2)))

def test_without_underscore():
    dicts = [{
        'uncertainty type': 4,
        'minimum': 2,
        'maximum': 5,
        'amount': 3,
        'loc': 3,
    }]
    answer = UncertaintyBase.from_dicts(*dicts)
    assert np.allclose(answer['uncertainty_type'][0], 4)
