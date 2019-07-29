from __future__ import division
from ..errors import InvalidParamsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from numpy import random, zeros, isnan, arange, linspace
from scipy import stats


class BetaUncertainty(UncertaintyBase):
    """
The Beta distribution has the probability distribution function:

.. math:: f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}(1 - x)^{\\beta - 1},

where the normalisation, *B*, is the beta function:

.. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}(1 - t)^{\\beta - 1} dt

The :math:`\\alpha` parameter is ``loc``, and :math:`\\beta` is ``shape``. By default, the Beta distribution is defined from 0 to 1; the lower and upper bounds can be rescaled with the ``minimum`` and ``maximum`` parameters.

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
        if ((params['minimum'] >= params['maximum']).sum() or
            (params['maximum'] <= params['minimum']).sum()):
            raise ImproperBoundsError("Min/max inconsistency.")

    @classmethod
    def _rescale(cls, params, results):
        mask = ~isnan(params['minimum'])
        params[~mask]['minimum'] = 0
        if mask.sum():
            results[mask] += params[mask]['minimum']
        mask = ~isnan(params['maximum'])
        params[~mask]['maximum'] = 1
        if mask.sum():
            results[mask] *= params[mask]['maximum']
        return results

    @classmethod
    def _loc_scale(cls, params):
        loc = params['minimum'].copy()
        loc[isnan(loc)] = 0
        scale = params['maximum'].copy()
        scale[isnan(scale)] = 1
        scale -= loc
        return loc, scale

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         transform=False):
        if not seeded_random:
            seeded_random = random
        # scale = params['scale']
        # scale[isnan(scale)] = 1
        return cls._rescale(
            params,
            # scale.reshape((-1, 1)) * seeded_random.beta(
            seeded_random.beta(
                params['loc'],
                params['shape'],
                size=(size, params.shape[0])).T
        )

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        loc, scale = cls._loc_scale(params)
        for row in range(params.shape[0]):
            results[row, :] = stats.beta.cdf(vector[row, :],
                                             params['loc'][row],
                                             params['shape'][row],
                                             loc=loc[row],
                                             scale=scale[row])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = zeros(percentages.shape)
        loc, scale = cls._loc_scale(params)
        for row in range(percentages.shape[0]):
            results[row, :] = stats.beta.ppf(percentages[row, :],
                                             params['loc'][row], params['shape'][row],
                                             loc=loc[row],
                                             scale=scale[row])
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        alpha = float(params['loc'])
        beta = float(params['shape'])
        loc = 0 if isnan(params['minimum']) else float(params['minimum'])
        scale = 1 if isnan(params['maximum']) else float(params['maximum'])
        # scale = 1 if isnan(params['maximum'])[0] else float(params['maximum'])
        if alpha <= 1 or beta <= 1:
            mode = "Undefined"
        else:
            mode = ((alpha - 1) / (alpha + beta - 2)) * scale + loc
        return {
            'mean': (alpha / (alpha + beta)) * scale + loc,
            'mode': mode,
            'median': "Not Implemented",
            'lower': "Not Implemented",
            'upper': "Not Implemented"
        }

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        loc = 0 if isnan(params['minimum']) else float(params['minimum'])
        scale = 1 if isnan(params['scale'])[0] else float(params['scale'])
        if xs is None:
            xs = linspace(loc, loc + scale, cls.default_number_points_in_pdf)
        ys = stats.beta.pdf(xs, params['loc'], params['shape'],
                            loc=loc, scale=scale)
        return xs, ys.reshape(ys.shape[1])
