# -*- coding: utf-8 -*
from ..errors import InvalidParamsError,\
    ImproperBoundsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from numpy import isnan, array, random, vstack
from scipy import stats


class DiscreteUniform(UncertaintyBase):
    """Discrete uniform distribution. In SciPy, the uniform distribution is defined from loc to loc+scale."""
    id = 7
    description = "Discrete uniform uncertainty"

    @classmethod
    def validate(cls, params):
        # No mean value
        if isnan(params['maximum']).sum():
            raise InvalidParamsError("Maximum values must always be defined.")
        # Minimum <= Maximum
        if (params['minimum'] >= params['maximum']).sum():
            raise ImproperBoundsError

    @classmethod
    def _fix_minimum(cls, params):
        mask = isnan(params['minimum'])
        params['minimum'][mask] = 0
        return params

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = random
        params = cls._fix_minimum(params)
        # randint has different behaviour than e.g. uniform. We can't pass in
        # arrays, but have to process them line by line.
        return vstack([seeded_random.randint(
            params['minimum'][i],  # Minimum (low)
            params['maximum'][i],  # Maximum (high)
            size=size
            ) for i in range(params.shape[0])])

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        params = cls._fix_minimum(params)
        for row in range(params.shape[0]):
            results[row, :] = stats.randint.cdf(vector[row, :],
                loc=params[row]['minimum'], scale=params[row]['maximum'] - \
                params[row]['minimum'])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        params = cls._fix_minimum(params)
        scale = (params['maximum'] - params['minimum']).reshape(
            params.shape[0], 1)
        return percentages * scale + params['minimum'].reshape(
            params.shape[0], 1)

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        params = cls._fix_minimum(params)
        mean = (params['maximum'] + params['minimum']) / 2
        return {'mean': mean, 'mode': mean, 'median': mean,
            'lower': params['minimum'], 'upper': params['maximum']}

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        params = cls._fix_minimum(params)
        if xs == None:
            xs = (params['minimum'], params['maximum'])
        percentage = 1 / (params['maximum'] - params['minimum'])
        ys = array([float(percentage) for x in xs])
        return array([float(x) for x in xs]), ys
