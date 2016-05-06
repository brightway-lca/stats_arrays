from __future__ import division
from .base import BoundedUncertaintyBase
from numpy import random, zeros, array


class BernoulliUncertainty(BoundedUncertaintyBase):
    id = 6
    description = u"Bernoulli uncertainty"

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        data = zeros((params.shape[0], size))
        if not seeded_random:
            seeded_random = random
        adjusted_means, scale = cls.rescale(params)
        adjusted_means.resize(params.shape[0], 1)
        # nums = seeded_random.random_sample(size * params.shape[0])
        # nums_rs = nums.reshape(params.shape[0], size)
        mask = seeded_random.random_sample(size * params.shape[0]).reshape(
            (params.shape[0], size)) >= adjusted_means
        data[mask] = 1
        return data

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        # Turn into column vector
        mean = array(params['loc'])
        mean.resize(params.shape[0], 1)
        return (vector >= mean) * 1

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        adjusted_means, scale = cls.rescale(params)
        length = params.shape[0]
        return (
            (percentages >= adjusted_means.reshape(length, 1))
            * scale.reshape(length, 1)
            + params['minimum'].reshape(length, 1)
        )
