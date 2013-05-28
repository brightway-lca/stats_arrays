from __future__ import division
from ..errors import InvalidParamsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from numpy import random, zeros, isnan, arange
from scipy import stats


class NormalUncertainty(UncertaintyBase):
    id = 3
    description = "Normal uncertainty"

    @classmethod
    def validate(cls, params):
        super(NormalUncertainty, cls).validate(params)
        if isnan(params['scale']).sum() or (params['scale'] <= 0).sum():
            raise InvalidParamsError("Real, positive scale values are required for normal uncertainties.")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = random
        return seeded_random.normal(
            params['loc'],
            params['scale'],
            size=(size, params.shape[0])).T

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.norm.cdf(vector[row, :],
                loc=params['loc'][row], scale=params['scale'][row])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = zeros(percentages.shape)
        for row in range(percentages.shape[0]):
            results[row, :] = stats.norm.ppf(percentages[row, :],
                loc=params['loc'][row], scale=params['scale'][row])
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        return {'mean': float(params['loc']), 'mode': float(params['loc']),
            'median': float(params['loc']),
            'lower': float(params['loc'] - 2 * params['scale']),
            'upper': float(params['loc'] + 2 * params['scale'])}

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        if xs == None:
            if isnan(params['minimum']):
                lower = params['loc'] - params['scale'] * \
                    cls.standard_deviations_in_default_range
            else:
                lower = params['minimum']
            if isnan(params['maximum']):
                upper = params['loc'] + params['scale'] * \
                    cls.standard_deviations_in_default_range
            else:
                upper = params['maximum']
            xs = arange(lower, upper, (upper - lower) / \
                cls.default_number_points_in_pdf)

        ys = stats.norm.pdf(xs, params['loc'], params['scale'])
        return xs, ys.reshape(ys.shape[1])
