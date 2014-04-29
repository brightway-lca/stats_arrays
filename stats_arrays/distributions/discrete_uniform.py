# -*- coding: utf-8 -*
from ..errors import InvalidParamsError,\
    ImproperBoundsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from scipy import stats
import numpy as np


class DiscreteUniform(UncertaintyBase):

    """
The discrete uniform distribution includes all integer values from the ``minimum`` up to, but excluding the ``maximum``.

See https://en.wikipedia.org/wiki/Uniform_distribution_(discrete).
    """
    id = 7
    description = "Discrete uniform uncertainty"

    @classmethod
    def validate(cls, params):
        # No mean value
        if np.isnan(params['maximum']).sum():
            raise InvalidParamsError("Maximum values must always be defined.")
        # Minimum <= Maximum
        if (params['minimum'] >= params['maximum']).sum():
            raise ImproperBoundsError

    @classmethod
    def fix_nan_minimum(cls, params):
        mask = np.isnan(params['minimum'])
        params['minimum'][mask] = 0
        return params

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = np.random
        params = cls.fix_nan_minimum(params)
        # randint has different behaviour than e.g. uniform. We can't pass in
        # arrays, but have to process them line by line.
        return np.vstack([
            seeded_random.randint(
                params['minimum'][i],  # Minimum (low)
                params['maximum'][i],  # Maximum (high)
                size=size
            ) for i in range(params.shape[0])
        ])

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        params = cls.fix_nan_minimum(params)
        for row in range(params.shape[0]):
            results[row, :] = stats.randint.cdf(
                vector[row, :],
                loc=params[row]['minimum'],
                scale=params[row]['maximum'] - params[row]['minimum']
            )
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        params = cls.fix_nan_minimum(params)
        scale = (params['maximum'] - params['minimum']).reshape(
            params.shape[0], 1)
        return percentages * scale + params['minimum'].reshape(
            params.shape[0], 1)

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        params = cls.fix_nan_minimum(params)
        mean = (params['maximum'] + params['minimum']) / 2
        return {
            'mean': mean,
            'mode': "Undefined",
            'median': int(round(mean)),
            'lower': params['minimum'],
            'upper': params['maximum']
        }

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        params = cls.fix_nan_minimum(params)
        if xs is None:
            xs = (params['minimum'], params['maximum'])
        percentage = 1 / (params['maximum'] - params['minimum'])
        ys = np.array([float(percentage) for x in xs])
        return np.array([float(x) for x in xs]), ys
