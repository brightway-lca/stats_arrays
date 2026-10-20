from typing import Any, Optional

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
        bad = np.isnan(params["loc"])
        if bad.any():
            raise InvalidParamsError(
                f"Real ``mu`` values required for generalized extreme value. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = np.isnan(params["scale"]) | (params["scale"] <= 0)
        if bad.any():
            raise InvalidParamsError(
                f"Real, positive ``sigma`` values required for generalized extreme value. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        bad = params["shape"] != 0
        if bad.any():
            raise InvalidParamsError(
                f"Non-zero ``xi`` values are not yet supported. "
                f"{cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
        data = seeded_random.gumbel(
            loc=params["loc"], scale=params["scale"], size=(size, params.shape[0])
        ).T
        return data
