from __future__ import print_function
from ...distributions import BernoulliUncertainty
from ..base import UncertaintyTestCase
import numpy as np


class BernoulliTestCase(UncertaintyTestCase):

    def test_bernoulli_ppf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        self.assertTrue(np.allclose(
            BernoulliUncertainty.ppf(
                oneDparams,
                np.array([[0, 0.25, 0.5, 0.75, 1]])
            ),
            np.array([[1, 1, 1, 4, 4]])
        ))
        self.assertTrue(np.allclose(
            BernoulliUncertainty.ppf(
                params,
                np.array([0.5, 0.8])
            ),
            np.array([[1], [4]])
        ))

    def test_bernoulli_cdf(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        self.assertTrue(np.allclose(
            BernoulliUncertainty.cdf(
                oneDparams,
                np.array([[1, 2, 3, 4]])
            ),
            np.array([[0, 0, 1, 1]])
        ))
        self.assertTrue(np.allclose(
            BernoulliUncertainty.cdf(
                params,
                np.array([1, 3])
            ),
            np.array([[0], [1]])
        ))

    def test_bernoulli_seeded_random(self):
        oneDparams = self.biased_params_1d()
        print(BernoulliUncertainty.random_variables(
            oneDparams,
            10,
            self.seeded_random()
        ))
        self.assertTrue(np.allclose(
            BernoulliUncertainty.random_variables(
                oneDparams,
                10,
                self.seeded_random()
            ),
            BernoulliUncertainty.random_variables(
                oneDparams,
                10,
                self.seeded_random()
            )
        ))

    def test_bernoulli_random(self):
        oneDparams = self.biased_params_1d()
        params = self.biased_params_2d()
        variables = BernoulliUncertainty.random_variables(oneDparams, 50000)
        self.assertTrue(0.3 < np.average(variables) < 0.35)
        self.assertEqual(variables.shape, (1, 50000))
        variables = BernoulliUncertainty.random_variables(params, 50000)
        self.assertTrue(0.3 < np.average(variables[0, :]) < 0.35)
        self.assertTrue(0.3 < np.average(variables[1, :]) < 0.35)
        self.assertEqual(variables.shape, (2, 50000))
        self.assertFalse(np.allclose(variables[0, :], variables[1,:]))

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
