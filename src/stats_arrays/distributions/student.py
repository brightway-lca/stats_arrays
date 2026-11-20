from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray, RandomSource


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
    def validate(cls, params: ParamsArray) -> None:
        bad = np.isnan(params["shape"]) | (params["shape"] <= 0)
        if bad.any():
            raise InvalidParamsError(
                f"Positive ``nu`` (degrees of freedom) values are required for Student's T. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = params["scale"] <= 0
        if bad.any():
            raise InvalidParamsError(
                f"Scale values, if specified, must be greater than zero for Student's T. "
                f"{cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[RandomSource] = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.default_rng()
        scale = params["scale"].copy()
        scale[np.isnan(scale)] = 1
        location = params["loc"].copy()
        location[np.isnan(location)] = 0
        data = (
            location.reshape((-1, 1))
            + scale.reshape((-1, 1))
            * seeded_random.standard_t(params["shape"], size=(size, params.shape[0])).T
        )
        return data

    # ``negative`` is ignored, as in ``random_variables``.

    @classmethod
    def _scipy_params(
        cls, params: ParamsArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return ``(df, loc, scale)`` as columns, with the defaults of
        ``random_variables`` for a NaN ``loc`` (0) or ``scale`` (1)."""
        loc = params["loc"].copy()
        loc[np.isnan(loc)] = 0
        scale = params["scale"].copy()
        scale[np.isnan(scale)] = 1
        return (
            params["shape"].reshape(-1, 1),
            loc.reshape(-1, 1),
            scale.reshape(-1, 1),
        )

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        from scipy import stats

        vector = cls.check_2d_inputs(params, vector)
        df, loc, scale = cls._scipy_params(params)
        return stats.t.cdf(vector, df, loc=loc, scale=scale)

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        from scipy import stats

        percentages = cls.check_2d_inputs(params, percentages)
        df, loc, scale = cls._scipy_params(params)
        return stats.t.ppf(percentages, df, loc=loc, scale=scale)
