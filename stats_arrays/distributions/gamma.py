from __future__ import division
from .base import UncertaintyBase
from ..errors import InvalidParamsError
import numpy as np


class GammaUncertainty(UncertaintyBase):
    """
The Gamma uncertainty distribution probability density function as a function of :math:`k`, the shape parameters, and :math:`\\theta`, the scale parameter:

.. math:: f(x;k,\\theta) =  \\frac{x^{k-1}e^{-\\frac{x}{\\theta}}}{\\theta^k\\Gamma(k)}

The scale parameter :math:`k` is ``shape``, and :math:`\\theta` is ``scale``. An optional location parameter, which offsets the distribution from the origin, can be specified in ``loc``.

See https://en.wikipedia.org/wiki/Gamma_distribution.
    """

    id = 9
    description = "Gamma uncertainty"

    @classmethod
    def validate(cls, params, transform=False):
        if (params['shape'] <= 0).sum() or np.isnan(params['shape']).sum():
            raise InvalidParamsError(
                "Positive shape (k) values required for Gamma distribution."
            )
        if (params['scale'] <= 0).sum() or np.isnan(params['scale']).sum():
            raise InvalidParamsError(
                "Positive scale (theta) values required for Gamma distribution."
            )

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         **kwargs):
        if seeded_random is None:
            seeded_random = np.random
        offset = params['loc'].copy()
        offset[np.isnan(offset)] = 0
        data = offset.reshape((-1, 1)) + seeded_random.gamma(
            shape=params['shape'],
            scale=params['scale'],
            size=(size, params.shape[0])).T
        data[params['negative'], :] = -1 * data[params['negative'], :]
        return data
