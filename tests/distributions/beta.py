from ..base import make_params_array as base_make
from scipy import stats
from stats_arrays.distributions import BetaUncertainty
from stats_arrays.errors import InvalidParamsError
import numpy as np
import pytest


ALPHA = 3.3
BETA = 2.2
INPUTS = np.array([0.5, 0.6, 0.8]).reshape((1, -1))
PDF = np.array([
    1.56479181717,
    1.82088038112,
    1.536047041126
])
CDF = np.array([
    0.30549,
    0.47638,
    0.8333
])


def make_params_array(length=2):
    params = base_make(length=length)
    params[:]['loc'] = ALPHA
    params[:]['shape'] = BETA
    return params

def test_random_variables_broadcasting():
    params = make_params_array()
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (2, 1000)
    assert 0.55 < np.average(results[0, :]) < 0.65
    assert 0.55 < np.average(results[1, :]) < 0.65

def test_random_variables_single_row():
    params = make_params_array(1)
    results = BetaUncertainty.random_variables(params, 1000)
    assert results.shape == (1, 1000)
    assert 0.55 < np.average(results) < 0.65

    # def test_random_variables_minimum(self):
    #     params = self.make_params_array()
    #     params['loc'] = 2
    #     params['shape'] = 5
    #     params['scale'] = 5
    #     results = BetaUncertainty.random_variables(params, 1000)
    #     self.assertEqual(results.shape, (1, 1000))
    #     self.assertTrue(0.26 * 5 < average(results) < 0.3 * 5)
    #     params = self.make_params_array(length=2)
    #     params[:]['loc'] = 2
    #     params[:]['shape'] = 5
    #     params[0]['scale'] = 5
    #     params[1]['scale'] = 10
    #     results = BetaUncertainty.random_variables(params, 1000)
    #     self.assertEqual(results.shape, (2, 1000))
    #     self.assertTrue(0.26 * 5 < average(results[0, :]) < 0.3 * 5)
    #     self.assertTrue(0.26 * 10 < average(results[1, :]) < 0.3 * 10)

def test_alpha_validation():
    params = make_params_array()
    params['loc'] = 0
    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)

def test_beta_validation():
    params = make_params_array()
    params['shape'] = 0
    with pytest.raises(InvalidParamsError):
        BetaUncertainty.validate(params)

    # def test_scale_valdiation(self):
    #     params = self.make_params_array()
    #     params['loc'] = 2
    #     params['shape'] = 5
    #     params['scale'] = 0
    #     self.assertRaises(InvalidParamsError,
    #                       BetaUncertainty.validate, params)
    #     params['scale'] = -1
    #     self.assertRaises(InvalidParamsError,
    #                       BetaUncertainty.validate, params)

def test_cdf():
    params = make_params_array(1)
    calculated = BetaUncertainty.cdf(params, INPUTS)
    assert np.allclose(CDF, calculated, rtol=1e-4)
    assert calculated.shape == (1, 3)

    # # def test_cdf_scaling(self):
    # #     params = self.make_params_array()
    # #     params['loc'] = 2
    # #     params['shape'] = 5
    # #     params['scale'] = 2
    # #     xs = arange(0.2, 2, 0.2).reshape((1, -1))
    # #     reference = stats.beta.cdf(xs, 2, 5, scale=2)
    # #     calculated = BetaUncertainty.cdf(params, xs)
    # #     self.assertTrue(allclose(reference, calculated))
    # #     self.assertEqual(reference.shape, calculated.shape)

def test_ppf():
    params = make_params_array(1)
    calculated = BetaUncertainty.ppf(params, CDF.reshape((1, -1)))
    assert np.allclose(INPUTS, calculated, rtol=1e-4)
    assert calculated.shape == (1, 3)

    # def test_ppf_scaling(self):
    #     params = self.make_params_array()
    #     params['loc'] = 2
    #     params['shape'] = 5
    #     params['minimum'] = 2
    #     xs = arange(0.1, 1, 0.1).reshape((1, -1))
    #     reference = stats.beta.ppf(xs, 2, 5, loc=2)
    #     calculated = BetaUncertainty.ppf(params, xs)
    #     self.assertTrue(allclose(reference, calculated))
    #     self.assertEqual(reference.shape, calculated.shape)

def test_pdf():
    params = make_params_array(1)
    calculated = BetaUncertainty.pdf(params, INPUTS)[1]
    assert np.allclose(PDF, calculated)
    assert calculated.shape == (3,)

    # def test_pdf_no_xs(self):
    #     params = self.make_params_array()
    #     params['loc'] = 2
    #     params['shape'] = 5
    #     xs = linspace(0, 1, 200)  # 200 is default number of points
    #     reference = stats.beta.pdf(xs, 2, 5)
    #     calculated = BetaUncertainty.pdf(params)
    #     self.assertTrue(allclose(reference, calculated[1]))
    #     self.assertEqual(reference.shape, calculated[1].shape)
    #     self.assertTrue(allclose(xs, calculated[0]))
    #     self.assertEqual(xs.shape, calculated[0].shape)
    #     self.assertEqual(calculated[1].shape, calculated[0].shape)

    # def test_pdf_scaling(self):
    #     params = self.make_params_array()
    #     params['loc'] = 2
    #     params['shape'] = 5
    #     params['scale'] = 2
    #     xs = arange(0.2, 2, 0.2)
    #     reference = stats.beta.pdf(xs, 2, 5, scale=2)
    #     calculated = BetaUncertainty.pdf(params, xs)
    #     self.assertTrue(allclose(reference, calculated[1]))
    #     self.assertEqual(reference.shape, calculated[1].shape)
    #     self.assertTrue(allclose(xs, calculated[0]))
    #     self.assertEqual(xs.shape, calculated[0].shape)
    #     self.assertEqual(calculated[1].shape, calculated[0].shape)

def test_seeded_random():
    sr = np.random.RandomState(111111)
    params = make_params_array(1)
    params['shape'] = params['loc'] = 1
    result = BetaUncertainty.random_variables(params, 4, seeded_random=sr)
    expected = np.array([0.59358266, 0.84368537, 0.01394206, 0.87557834])
    assert np.allclose(result, expected)
