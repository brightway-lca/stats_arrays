from ...errors import InvalidParamsError
from ...distributions import LognormalUncertainty as LU
from ..base import UncertaintyTestCase
from scipy.special import erf
import numpy as np


class LognormalTestCase(UncertaintyTestCase):
    def pdf(self, x, mu, sigma):
        return 1 / (x * np.sqrt(2 * np.pi * sigma ** 2)
            ) * np.e ** (-((np.log(x) - mu) ** 2) / (
            2 * sigma ** 2))

    def cdf(self, x, mu, sigma):
        return 0.5 * (1 + erf((np.log(x) - mu) / np.sqrt(
            2 * sigma ** 2)))

    def test_pdf_positive(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma})
        self.assertTrue(np.allclose(
            self.pdf(1.2, mu, sigma),
            # [0] are X values, [1] are Y values
            LU.pdf(pa, np.array((1.2,)))[1]
        ))

    def test_pdf_negative(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma, 'negative': True})
        self.assertTrue(np.allclose(
            self.pdf(1.2, mu, sigma),
            # [0] are X values, [1] are Y values
            LU.pdf(pa, np.array((-1.2,)))[1]
        ))

    def test_cdf_positive(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma})
        self.assertTrue(np.allclose(
            self.cdf(1.2, mu, sigma),
            LU.cdf(pa, np.array((1.2,)))[0]
        ))

    def test_cdf_negative(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma, 'negative': True})
        self.assertTrue(np.allclose(
            self.cdf(1.2, mu, sigma),
            LU.cdf(pa, np.array((-1.2,)))[0]
        ))

    def test_cdf_multirow(self):
        pa = LU.from_dicts(
            {'loc': 0.4, 'scale': 0.1, 'negative': True},
            {'loc': 0.6, 'scale': 0.2, 'negative': False}
        )
        self.assertTrue(np.allclose(
            self.cdf(
                np.array((1.2, 1.5)),
                np.array((0.4, 0.6)),
                np.array((0.1, 0.2))
            ),
            LU.cdf(pa, np.array((-1.2, 1.5))).ravel()
        ))

    def test_ppf_positive(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma})
        cdf = self.cdf(1.3, mu, sigma)
        self.assertTrue(np.allclose(
            1.3,
            LU.ppf(pa, np.array((cdf,)))[0]
        ))

    def test_ppf_negative(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma, 'negative': True})
        cdf = self.cdf(1.3, mu, sigma)
        self.assertTrue(np.allclose(
            -1.3,
            LU.ppf(pa, np.array((cdf,)))[0]
        ))

    def test_validation(self):
        dicts = [
            {'loc': np.NaN, 'scale': 0.1},
            {'loc': 0.1, 'scale': np.NaN},
        ]
        for d in dicts:
            with self.assertRaises(InvalidParamsError):
                LU.validate(LU.from_dicts(d))

    def test_seeded_random(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma})
        self.assertTrue(np.allclose(
            LU.random_variables(pa, 100, self.seeded_random()),
            LU.random_variables(pa, 100, self.seeded_random())
        ))

    def test_rng(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        median = np.exp(mu)
        ci_upper_975 = np.exp(mu) * (np.exp(sigma) ** 1.96)
        pa = LU.from_dicts({'loc': mu, 'scale': sigma})
        sample = LU.random_variables(pa, size=1e5).ravel()
        sample.sort()
        self.assertTrue(np.allclose(np.median(sample), median, 0.01))
        self.assertTrue(np.allclose(sample[97500], ci_upper_975, 0.01))

    def test_rng_negative(self):
        mu, sigma = np.random.random() / 5 + 0.2, np.random.random() / 10 + 0.1
        pa = LU.from_dicts({'loc': mu, 'scale': sigma, 'negative': True})
        sample = LU.random_variables(pa, size=100).ravel()
        self.assertEqual((sample < 0).sum(), 100)

#     def test_lognormal_statistics(self):
#         oneDparams = self.biased_params_1d()
#         oneDparams['scale'] = 0.8
#         to_array = lambda x: np.array([x[key] for key in sorted(x.keys())])
#         self.assertTrue(np.allclose(
#             to_array({
#                 'upper': 14.859097922170479,
#                 'lower': 0.60568955155650162,
#                 'median': 3.0000000595022631,
#                 'mode': 1.5818772733323232,
#                 'mean': 4.131383414350033
#             }),
#             to_array(LognormalUncertainty.statistics(oneDparams))
#         ))

#     def test_lognormal_negative_statistics(self):
#         oneDparams = self.biased_params_1d()
#         oneDparams['scale'] = 0.8
#         oneDparams['minimum'] = -4
#         oneDparams['loc'] = -3
#         oneDparams['maximum'] = -1
#         to_array = lambda x: np.array([x[key] for key in sorted(x.keys())])
#         self.assertTrue(np.allclose(
#             to_array({
#                 'lower': -14.859097922170479,
#                 'upper': -0.60568955155650162,
#                 'median': -3.0000000595022631,
#                 'mode': -1.5818772733323232,
#                 'mean': -4.131383414350033
#             }),
#             to_array(LognormalUncertainty.statistics(oneDparams, transform=True))
#         ))
