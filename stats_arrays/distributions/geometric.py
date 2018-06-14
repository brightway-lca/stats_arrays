from __future__ import division
from ..utils import one_row_params_array
from ..errors import ImproperBoundsError
from .base import BoundedUncertaintyBase
from scipy import stats
import numpy as np


class UniformUncertainty(BoundedUncertaintyBase):

    """Continuous uniform distribution. In SciPy, the uniform distribution is defined from loc to loc+scale."""
    id = 4
    description = "Uniform uncertainty"

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = np.random
        return seeded_random.uniform(
            params['minimum'],  # Minimum (low)
            params['maximum'],  # Maximum (high)
            size=(size, params.shape[0])).T

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.uniform.cdf(vector[row, :],
                                                loc=params[row]['minimum'], scale=params[row]['maximum'] -
                                                params[row]['minimum'])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        scale = (params['maximum'] - params['minimum']).reshape(
            params.shape[0], 1)
        return percentages * scale + params['minimum'].reshape(
            params.shape[0], 1)

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        mean = (params['maximum'] + params['minimum']) / 2
        return {'mean': mean, 'mode': mean, 'median': mean,
                'lower': params['minimum'], 'upper': params['maximum']}

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        if xs is None:
            xs = (params['minimum'], params['maximum'])
        percentage = 1 / (params['maximum'] - params['minimum'])
        ys = np.array([float(percentage) for x in xs])
        return np.array([float(x) for x in xs]), ys


class TriangularUncertainty(BoundedUncertaintyBase):
    id = 5
    description = "Triangular uncertainty"

    @classmethod
    def validate(cls, params):
        super(TriangularUncertainty, cls).validate(params)
        if ((params['loc'] > params['maximum']).sum() or
            (params['loc'] < params['minimum']).sum()):
            raise ImproperBoundsError("Most likely value outside the given bounds.")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = np.random
        return seeded_random.triangular(
            params['minimum'],  # Left
            params['loc'],  # Mode
            params['maximum'],  # Right
            size=(size, params.shape[0])).T

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        # Adjust parameters to (0,1) range
        adjusted_means, scale = cls.rescale(params)
        # To be broadcasted correctly, scale and mins must be column vectors
        scale.resize(scale.shape[0], 1)
        mins = np.array(params['minimum'])
        mins.resize(params.shape[0], 1)
        # Adjust values to use cdf for to (0,1) range
        adjusted_vector = (vector - mins) / scale
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.triang.cdf(adjusted_vector[row, :],
                                               adjusted_means[row])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        adjusted_means, scale = cls.rescale(params)
        scale.resize(scale.shape[0], 1)
        adjusted_means.resize(scale.shape[0], 1)
        mins = np.array(params['minimum'])
        mins.resize(params.shape[0], 1)
        return stats.triang.ppf(percentages, adjusted_means) * scale + mins

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        mode = params['loc']
        mean = (params['minimum'] + params['maximum'] + mode) / 3
        lower, median, upper = cls.ppf(params, np.array([[0.0125, 0.5, 0.9875]])).ravel()
        return {'mean': float(mean), 'median': float(median),
                'mode': float(mode), 'lower': float(lower),
                'upper': float(upper)}

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        if xs is None:
            lower = params['minimum']
            upper = params['maximum']
            mode = params['loc']
            if not mode:
                mode = (upper + lower) / 2
            xs = np.array([float(x) for x in (lower, mode, upper)])
            ys = np.array([0, float((mode - lower) / (upper - lower)), 0])
        else:
            adjusted_means, scale = cls.rescale(params)
            adj_xs = (xs - params['minimum']) / scale
            ys = stats.triang.pdf(adj_xs, adjusted_means)
        return xs, ys
