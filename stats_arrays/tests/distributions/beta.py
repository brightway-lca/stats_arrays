from __future__ import division
from numpy import *
from scipy import stats
from ...distributions import BetaUncertainty
from ...errors import InvalidParamsError
from ..base import UncertaintyTestCase


class BetaTestCase(UncertaintyTestCase):

    def test_random_variables_broadcasting(self):
        params = self.make_params_array(length=2)
        params[:]['loc'] = 2
        params[:]['shape'] = 5
        results = BetaUncertainty.random_variables(params, 1000)
        self.assertEqual(results.shape, (2, 1000))
        self.assertTrue(0.26 < average(results[0, :]) < 0.3)
        self.assertTrue(0.26 < average(results[1, :]) < 0.3)

    def test_random_variables(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        results = BetaUncertainty.random_variables(params, 1000)
        self.assertEqual(results.shape, (1, 1000))
        self.assertTrue(0.26 < average(results) < 0.3)

    def test_random_variables_scaling(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        params['scale'] = 5
        results = BetaUncertainty.random_variables(params, 1000)
        self.assertEqual(results.shape, (1, 1000))
        self.assertTrue(0.26 * 5 < average(results) < 0.3 * 5)
        params = self.make_params_array(length=2)
        params[:]['loc'] = 2
        params[:]['shape'] = 5
        params[0]['scale'] = 5
        params[1]['scale'] = 10
        results = BetaUncertainty.random_variables(params, 1000)
        self.assertEqual(results.shape, (2, 1000))
        self.assertTrue(0.26 * 5 < average(results[0, :]) < 0.3 * 5)
        self.assertTrue(0.26 * 10 < average(results[1, :]) < 0.3 * 10)

    def test_alpha_validation(self):
        params = self.make_params_array()
        params['loc'] = 0
        params['shape'] = 5
        self.assertRaises(InvalidParamsError,
                          BetaUncertainty.validate, params)

    def test_beta_validation(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 0
        self.assertRaises(InvalidParamsError,
                          BetaUncertainty.validate, params)

    def test_scale_valdiation(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        params['scale'] = 0
        self.assertRaises(InvalidParamsError,
                          BetaUncertainty.validate, params)
        params['scale'] = -1
        self.assertRaises(InvalidParamsError,
                          BetaUncertainty.validate, params)

    def test_cdf(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        xs = arange(0.1, 1, 0.1).reshape((1, -1))
        reference = stats.beta.cdf(xs, 2, 5)
        calculated = BetaUncertainty.cdf(params, xs)
        self.assertTrue(allclose(reference, calculated))
        self.assertEqual(reference.shape, calculated.shape)

    def test_cdf_scaling(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        params['scale'] = 2
        xs = arange(0.2, 2, 0.2).reshape((1, -1))
        reference = stats.beta.cdf(xs, 2, 5, scale=2)
        calculated = BetaUncertainty.cdf(params, xs)
        self.assertTrue(allclose(reference, calculated))
        self.assertEqual(reference.shape, calculated.shape)

    def test_ppf(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        xs = arange(0.1, 1, 0.1).reshape((1, -1))
        reference = stats.beta.ppf(xs, 2, 5)
        calculated = BetaUncertainty.ppf(params, xs)
        self.assertTrue(allclose(reference, calculated))
        self.assertEqual(reference.shape, calculated.shape)

    def test_ppf_scaling(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        params['scale'] = 2
        xs = arange(0.1, 1, 0.1).reshape((1, -1))
        reference = stats.beta.ppf(xs, 2, 5, scale=2)
        calculated = BetaUncertainty.ppf(params, xs)
        self.assertTrue(allclose(reference, calculated))
        self.assertEqual(reference.shape, calculated.shape)

    def test_pdf(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        xs = arange(0.1, 1, 0.1)
        reference = stats.beta.pdf(xs, 2, 5)
        calculated = BetaUncertainty.pdf(params, xs)
        self.assertTrue(allclose(reference, calculated[1]))
        self.assertEqual(reference.shape, calculated[1].shape)
        self.assertTrue(allclose(xs, calculated[0]))
        self.assertEqual(xs.shape, calculated[0].shape)
        self.assertEqual(calculated[1].shape, calculated[0].shape)

    def test_pdf_no_xs(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        xs = arange(0, 1, 1. / 200)  # 200 is default number of points
        reference = stats.beta.pdf(xs, 2, 5)
        calculated = BetaUncertainty.pdf(params)
        self.assertTrue(allclose(reference, calculated[1]))
        self.assertEqual(reference.shape, calculated[1].shape)
        self.assertTrue(allclose(xs, calculated[0]))
        self.assertEqual(xs.shape, calculated[0].shape)
        self.assertEqual(calculated[1].shape, calculated[0].shape)

    def test_pdf_scaling(self):
        params = self.make_params_array()
        params['loc'] = 2
        params['shape'] = 5
        params['scale'] = 2
        xs = arange(0.2, 2, 0.2)
        reference = stats.beta.pdf(xs, 2, 5, scale=2)
        calculated = BetaUncertainty.pdf(params, xs)
        self.assertTrue(allclose(reference, calculated[1]))
        self.assertEqual(reference.shape, calculated[1].shape)
        self.assertTrue(allclose(xs, calculated[0]))
        self.assertEqual(xs.shape, calculated[0].shape)
        self.assertEqual(calculated[1].shape, calculated[0].shape)

    def test_seeded_random(self):
        sr = self.seeded_random()
        params = self.make_params_array()
        params['shape'] = params['loc'] = 1
        self.assertTrue(allclose(
            BetaUncertainty.random_variables(params, 4, seeded_random=sr),
            array([0.59358266, 0.84368537, 0.01394206, 0.87557834])
        ))
