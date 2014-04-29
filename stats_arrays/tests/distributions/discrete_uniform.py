from __future__ import division
from ...distributions import DiscreteUniform
from ..base import UncertaintyTestCase
import numpy as np


class DiscreteUniformTestCase(UncertaintyTestCase):

    def test_array_shape_1d(self):
        params = self.make_params_array(length=1)
        params['minimum'] = 0
        params['maximum'] = 10
        sample = DiscreteUniform.random_variables(params, 100)
        self.assertEqual(sample.shape, (1, 100))

    def test_array_shape_2d(self):
        params = self.make_params_array(length=10)
        params['minimum'] = 0
        params['maximum'] = 10
        sample = DiscreteUniform.random_variables(params, 100)
        self.assertEqual(sample.shape, (10, 100))

    def test_random_variables(self):
        params = self.make_params_array(length=10)
        params['minimum'] = 5
        params['maximum'] = 10
        sample = DiscreteUniform.random_variables(params, 10000)
        self.assertEqual([5, 6, 7, 8, 9], np.unique(sample).tolist())
