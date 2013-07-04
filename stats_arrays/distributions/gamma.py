from __future__ import division
from .base import UncertaintyBase
import numpy as np


class GammaUncertainty(UncertaintyBase):
    """
Gamma distribution.

shape : scalar > 0
    The shape of the gamma distribution.
    Called "amount" in params array.
scale : scalar > 0, optional
    The scale of the gamma distribution.  Default is equal to 1.
    Called "sigma" in params array.

    """

    id = 21
    description = "Gamma uncertainty"

    @classmethod
    def validate(cls, params, transform=False):
        return

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         **kwargs):
        if not seeded_random:
            seeded_random = np.random

        data = seeded_random.gamma(
            shape=params['shape'],
            scale=params['scale'],
            size=(size, params.shape[0])).T
        data[params['negative'], :] = -1 * data[params['negative'], :]
        return data
