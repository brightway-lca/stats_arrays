from __future__ import division
from ..errors import UndefinedDistributionError
from .base import UncertaintyBase
from numpy import repeat, tile


class UndefinedUncertainty(UncertaintyBase):

    """Undefined or unknown uncertainty"""
    id = 0
    description = "Undefined or unknown uncertainty"

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        return repeat(params['loc'], size).reshape((params.shape[0],
                                                    size))

    @classmethod
    def cdf(cls, params, vector):
        raise UndefinedDistributionError(
            "Can't calculate percentages for an undefined distribution.")

    @classmethod
    def ppf(cls, params, percentages):
        return tile(params['loc'].reshape((params.shape[0], 1)),
                    percentages.shape[1])


class NoUncertainty(UndefinedUncertainty):
    id = 1
    description = "No uncertainty"
