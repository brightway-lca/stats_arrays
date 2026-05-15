from typing import Optional

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray


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
        seeded_random: Optional[np.random.RandomState] = None,
        **kwargs,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
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
