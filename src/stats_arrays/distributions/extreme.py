from typing import Optional

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray


class GeneralizedExtremeValueUncertainty(UncertaintyBase):
    """
    The generalized extreme value uncertainty, or Fisher-Tippett, distribution is described in the Wikipedia article: http://en.wikipedia.org/wiki/Generalized_extreme_value_distribution.

    In our implementation, :math:`\\mu` is ``location``, :math:`\\sigma` is ``scale``, and :math:`\\xi`  is ``shape``.

    """

    id = 11
    description = "Generalized extreme value uncertainty"

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        if np.isnan(params["loc"]).sum():
            raise InvalidParamsError(
                "Real ``mu`` values needed for generalized extreme value."
            )
        if (params["scale"] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive ``sigma`` values need for generalized extreme value."
            )
        if (params["shape"] != 0).sum():
            raise InvalidParamsError("Non-zero ``xi`` values are not yet supported.")

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
        data = seeded_random.gumbel(
            loc=params["loc"], scale=params["scale"], size=(size, params.shape[0])
        ).T
        return data
