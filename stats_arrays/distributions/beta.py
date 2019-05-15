from __future__ import division
from ..errors import InvalidParamsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from numpy import random, zeros, isnan, arange
from scipy import stats


class BetaUncertainty(UncertaintyBase):
    """
The Beta distribution has the probability distribution function:

.. math:: f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}(1 - x)^{\\beta - 1},

where the normalisation, *B*, is the beta function:

.. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}(1 - t)^{\\beta - 1} dt

The :math:`\\alpha` parameter is ``loc``, and :math:`\\beta` is ``shape``. By default, the Beta distribution is defined from 0 to 1; the upper bound can be rescaled with the ``scale`` parameter.

Wikipedia: `Beta distribution <http://en.wikipedia.org/wiki/Beta_distribution>`_
    """
    id = 10
    description = "Beta uncertainty"

    @classmethod
    def validate(cls, params):
        if (params['loc'] > 0).sum() != params.shape[0]:
            raise InvalidParamsError("Real, positive alpha values are" +
                                     " required for Beta uncertainties.")
        if (params['shape'] > 0).sum() != params.shape[0]:
            raise InvalidParamsError("Real, positive beta values are" +
                                     " required for Beta uncertainties.")
        if (params['scale'] <= 0).sum():
            raise InvalidParamsError("Scale value must be positive or NaN")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         transform=False):
        if not seeded_random:
            seeded_random = random
        scale = params['scale']
        scale[isnan(scale)] = 1
        return scale.reshape((-1, 1)) * seeded_random.beta(
            params['loc'],
            params['shape'],
            size=(size, params.shape[0])).T

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        scale = params['scale']
        scale[isnan(scale)] = 1
        for row in range(params.shape[0]):
            results[row, :] = stats.beta.cdf(vector[row, :],
                                             params['loc'][row], params['shape'][row],
                                             scale=scale[row])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = zeros(percentages.shape)
        scale = params['scale']
        scale[isnan(scale)] = 1
        for row in range(percentages.shape[0]):
            results[row, :] = stats.beta.ppf(percentages[row, :],
                                             params['loc'][row], params['shape'][row],
                                             scale=scale[row])
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        alpha = float(params['loc'])
        beta = float(params['shape'])
        # scale = 1 if isnan(params['maximum'])[0] else float(params['maximum'])
        if alpha <= 1 or beta <= 1:
            mode = "Undefined"
        else:
            mode = (alpha - 1) / (alpha + beta - 2)
        return {
            'mean': alpha / (alpha + beta),
            'mode': mode,
            'median': "Not Implemented",
            'lower': "Not Implemented",
            'upper': "Not Implemented"
        }

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        scale = 1 if isnan(params['scale'])[0] else float(params['scale'])
        if xs is None:
            xs = arange(0, scale, scale / cls.default_number_points_in_pdf)
        ys = stats.beta.pdf(xs, params['loc'], params['shape'],
                            scale=scale)
        return xs, ys.reshape(ys.shape[1])
