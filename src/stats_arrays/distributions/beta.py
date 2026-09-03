from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError
from stats_arrays.utils import (
    ParamsArray,
    StatisticsResult,
    one_row_params_array,
    rescale_vector_to_params,
)


class BetaUncertainty(UncertaintyBase):
    """
    The Beta distribution has the probability distribution function:

    .. math:: f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}(1 - x)^{\\beta - 1},

    where the normalisation, *B*, is the beta function:

    .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}(1 - t)^{\\beta - 1} dt

    The :math:`\\alpha` parameter is ``loc``, and :math:`\\beta` is ``shape``. By default, the Beta distribution is defined from 0 to 1; the lower and upper bounds can be rescaled with the ``minimum`` and ``maximum`` parameters.

    Wikipedia: `Beta distribution <http://en.wikipedia.org/wiki/Beta_distribution>`_
    """

    id = 10
    description = "Beta uncertainty"

    @classmethod
    def _safe_loc(cls, params: ParamsArray) -> npt.NDArray:
        """Get `loc` in the form needed for Scipy functions, handling `nan` values."""
        loc = params["minimum"].copy()
        loc[np.isnan(loc)] = 0
        return loc

    @classmethod
    def _safe_scale(cls, params: ParamsArray) -> npt.NDArray:
        """Get `scale` in the form needed for Scipy functions, handling `nan` values."""
        min_ = params["minimum"].copy()
        max_ = params["maximum"].copy()
        min_[np.isnan(min_)] = 0
        max_[np.isnan(max_)] = 1
        return max_ - min_

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        bad = ~(params["loc"] > 0)
        if bad.any():
            raise InvalidParamsError(
                f"Real, positive alpha values are required for Beta uncertainties. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = ~(params["shape"] > 0)
        if bad.any():
            raise InvalidParamsError(
                f"Real, positive beta values are required for Beta uncertainties. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = params["minimum"] >= params["maximum"]
        if bad.any():
            raise ImproperBoundsError(
                f"Min/max inconsistency. {cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
        transform: bool = False,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
        return rescale_vector_to_params(
            params=params,
            vector=seeded_random.beta(
                params["loc"], params["shape"], size=(size, params.shape[0])
            ).T,
        )

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        loc, scale = cls._safe_loc(params), cls._safe_scale(params)
        for index, _ in enumerate(params):
            results[index, :] = stats.beta.cdf(
                vector[index, :],
                params["loc"][index],
                params["shape"][index],
                loc=loc[index],
                scale=scale[index],
            )
        return results

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        results = np.zeros(percentages.shape)
        loc, scale = cls._safe_loc(params), cls._safe_scale(params)
        for index, _ in enumerate(percentages):
            results[index, :] = stats.beta.ppf(
                percentages[index, :],
                params["loc"][index],
                params["shape"][index],
                loc=loc[index],
                scale=scale[index],
            )
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> StatisticsResult:
        alpha, beta = float(params["loc"].flat[0]), float(params["shape"].flat[0])
        minimum = float(cls._safe_loc(params).flat[0])
        scale = float(cls._safe_scale(params).flat[0])

        if alpha > 1 and beta > 1:
            mode = ((alpha - 1) / (alpha + beta - 2)) * scale + minimum
        elif alpha <= 1 < beta:
            mode = minimum
        elif beta <= 1 < alpha:
            mode = minimum + scale
        else:
            # alpha = beta = 1 is flat; alpha, beta < 1 is bimodal at the bounds
            mode = None

        lower, median, upper = cls.ppf(params, np.array([[0.025, 0.5, 0.975]])).ravel()
        return {
            "mean": (alpha / (alpha + beta)) * scale + minimum,
            "mode": mode,
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
        }

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ParamsArray, xs: Optional[npt.NDArray] = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        loc = float(cls._safe_loc(params)[0][0])
        scale = float(cls._safe_scale(params)[0][0])

        if xs is None:
            xs = np.linspace(loc, loc + scale, cls.default_number_points_in_pdf)
        ys = stats.beta.pdf(xs, params["loc"], params["shape"], loc=loc, scale=scale)
        return xs, ys.reshape(ys.shape[1])
