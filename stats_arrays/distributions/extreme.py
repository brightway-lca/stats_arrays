from __future__ import division
from base import UncertaintyBase
from ..errors import InvalidParamsError
import numpy as np


class GeneralizedExtremeValueUncertainty(UncertaintyBase):
    r"""
The generalized extreme value uncertainty distribution has the cumulative distribution function (the PDF is even more complicated):

.. math:: f(x;\mu,\sigma,\xi) = \exp\left\{-\left[1+\xi\left(\frac{x-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}

In our implementation, :math:`\mu` is ``location``, :math:`\sigma` is ``scale``, and :math:`\xi`  is ``shape``.

See http://en.wikipedia.org/wiki/Generalized_extreme_value_distribution.
    """
    id = 11
    description = "Generalized extreme value uncertainty"

    @classmethod
    def validate(cls, params):
        if np.isnan(params['loc']).sum():
            raise InvalidParamsError(
                "Real ``mu`` values needed for generalized extreme value.")
        if (params['scale'] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive ``sigma`` values need for generalized extreme value.")
        if (params['shape'] != 0).sum():
            raise InvalidParamsError(
                "Non-zero ``xi`` values are not yet supported.")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None, **kwargs):
        if seeded_random is None:
            seeded_random = np.random
        data = seeded_random.gumbel(
            loc=params['loc'],
            scale=params['scale'],
            size=(size, params.shape[0])
        ).T
        return data
