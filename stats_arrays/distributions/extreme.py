from __future__ import division
from .base import UncertaintyBase
from ..errors import InvalidParamsError
import numpy as np


class GeneralizedExtremeValueUncertainty(UncertaintyBase):
    """
The generalized extreme value uncertainty, or Fisher-Tippett, distribution is described in the Wikipedia article: http://en.wikipedia.org/wiki/Generalized_extreme_value_distribution.

In our implementation, :math:`\\mu` is ``location``, :math:`\\sigma` is ``scale``, and :math:`\\xi`  is ``shape``.

    """
    id = 11
    description = "Generalized extreme value uncertainty"

    @classmethod
    def validate(cls, params):
        if np.isnan(params['loc']).sum():
            raise InvalidParamsError(
                u"Real ``mu`` values needed for generalized extreme value.")
        if (params['scale'] <= 0).sum():
            raise InvalidParamsError(
                u"Real, positive ``sigma`` values need for generalized extreme value.")
        if (params['shape'] != 0).sum():
            raise InvalidParamsError(
                u"Non-zero ``xi`` values are not yet supported.")

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
