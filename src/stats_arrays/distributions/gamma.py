from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray, RandomSource


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
    def validate(cls, params: ParamsArray, transform: bool = False) -> None:
        cls._check_finite(params, "loc", "scale", "shape")
        bad = np.isnan(params["shape"]) | (params["shape"] <= 0)
        if bad.any():
            raise InvalidParamsError(
                f"Positive shape (k) values required for Gamma distribution. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = np.isnan(params["scale"]) | (params["scale"] <= 0)
        if bad.any():
            raise InvalidParamsError(
                f"Positive scale (theta) values required for Gamma distribution. "
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
        offset = params["loc"].copy()
        offset[np.isnan(offset)] = 0
        data = (
            offset.reshape((-1, 1))
            + seeded_random.gamma(
                shape=params["shape"],
                scale=params["scale"],
                size=(size, params.shape[0]),
            ).T
        )
        data[params["negative"], :] = -1 * data[params["negative"], :]
        return data

    # ``loc``, ``scale`` and ``shape`` describe the unsigned values, and
    # ``random_variables`` negates rows with ``negative`` set. As in
    # ``LognormalUncertainty``, ``ppf`` negates those rows too, and ``cdf``
    # evaluates the unsigned distribution at ``-x`` for them, so that it
    # inverts ``ppf``.

    @classmethod
    def _scipy_params(
        cls, params: ParamsArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return ``(shape, loc, scale)`` as columns, with a NaN ``loc`` as 0."""
        loc = params["loc"].copy()
        loc[np.isnan(loc)] = 0
        return (
            params["shape"].reshape(-1, 1),
            loc.reshape(-1, 1),
            params["scale"].reshape(-1, 1),
        )

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        """Return the cumulative probabilities of ``vector``, with ``params``
        rows and ``vector`` columns.

        .. important:: For rows with ``negative`` set this is the cumulative
           probability of the *unsigned* distribution at ``-x``, which is
           P(X >= x) for the signed values X and decreases as x increases. It
           is the inverse of :meth:`ppf`. ``LognormalUncertainty.cdf`` follows
           the same convention.
        """
        from scipy import stats

        vector = cls.check_2d_inputs(params, vector)
        unsigned = np.where(params["negative"].reshape(-1, 1), -vector, vector)
        shape, loc, scale = cls._scipy_params(params)
        return stats.gamma.cdf(unsigned, shape, loc=loc, scale=scale)

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        """Return the values at ``percentages``, with ``params`` rows and
        ``percentages`` columns.

        .. important:: For rows with ``negative`` set, the quantile of the
           unsigned distribution is negated, as ``random_variables`` negates
           its draws. Percentage ``p`` therefore gives the ``1 - p`` quantile
           of the signed values, and the row decreases as ``p`` increases.
        """
        from scipy import stats

        percentages = cls.check_2d_inputs(params, percentages)
        shape, loc, scale = cls._scipy_params(params)
        results = stats.gamma.ppf(percentages, shape, loc=loc, scale=scale)
        results[params["negative"]] *= -1
        return results
