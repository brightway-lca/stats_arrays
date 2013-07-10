from __future__ import division
from base import UncertaintyBase
from stats_toolkit.errors import InvalidParamsError
import numpy as np


class WeibullUncertainty(UncertaintyBase):
    """
The Weibull distribution has the probability distribution function:

.. math:: f(x; k, \lambda) = \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k - 1} e^{- \left( x / \lambda \right)^{k}}

In our implementation, :math:`\lambda` is ``scale``, and :math:`k`  is ``shape``.

See http://en.wikipedia.org/wiki/Weibull_distribution.
    """
    id = 22
    description = "Weibull uncertainty"

    @classmethod
    def validate(cls, params):
        if (params['shape'] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive ``k`` values need for Weibull.")
        if (params['scale'] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive ``lambda`` values need for Weibull.")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         transform=False):
        if seeded_random is None:
            seeded_random = np.random
        data = params['scale'].reshape((-1, 1)) * seeded_random.weibull(
            params['shape'],
            size=(size, params.shape[0])
        ).T
        data[params['negative'], :] = -1 * data[params['negative'], :]
        return data
