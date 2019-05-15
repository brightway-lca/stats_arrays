from __future__ import division
from .base import UncertaintyBase
from ..errors import InvalidParamsError
import numpy as np


class StudentsTUncertainty(UncertaintyBase):
    """
The Student's T uncertainty distribution probability density function is a function of :math:`\\nu`, the degrees of freedom:

.. math:: f(x; \\nu) = \\frac{\\Gamma(\\frac{\\nu+1}{2})} {\\sqrt{\\nu\\pi},\\Gamma(\\frac{\\nu}{2})} \\left(1+\\frac{x^2}{\\nu} \\right)^{-\\frac{\\nu+1}{2}}

A non-standardized distribution, with a location and scale parameter, is also possible, through the transformation:

.. math:: X = \\mu + \\sigma f

In our implementation, the location parameter :math:`\\mu` is ``location``, the scale parameter :math:`\\sigma` is ``scale``, and :math:`\\nu` (the degrees of freedom)  is ``shape``.

See http://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    id = 12
    description = "Student's T uncertainty"

    @classmethod
    def validate(cls, params):
        if (params['shape'] <= 0).sum() or np.isnan(params['shape']).sum():
            raise InvalidParamsError(
                "Positive ``nu`` (degrees of freedom) values are required for"
                " Student's T."
            )
        if (params['scale'] <= 0).sum():
            raise InvalidParamsError(
                "Scale values, if specified, must be greater than zero for"
                " Student's T."
            )

    @classmethod
    def random_variables(cls, params, size, seeded_random=None, **kwargs):
        if seeded_random is None:
            seeded_random = np.random
        scale = params['scale'].copy()
        scale[np.isnan(scale)] = 1
        location = params['loc'].copy()
        location[np.isnan(location)] = 0
        data = location + scale * seeded_random.standard_t(
            params['shape'],
            size=(size, params.shape[0])
        ).T
        return data
