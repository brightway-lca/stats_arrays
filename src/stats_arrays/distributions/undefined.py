from typing import Optional

import numpy as np
import numpy.typing as npt
from numpy import repeat, tile

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import UndefinedDistributionError
from stats_arrays.utils import ParamsArray, RandomSource


class UndefinedUncertainty(UncertaintyBase):
    """Undefined or unknown uncertainty.

    Every sample is ``loc``. ``validate`` does not check ``loc``, so an
    infinite or NaN ``loc`` is returned as it is."""

    id = 0
    description = "Undefined or unknown uncertainty"

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[RandomSource] = None,
    ) -> npt.NDArray:
        return repeat(params["loc"], size).reshape((params.shape[0], size))

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        raise UndefinedDistributionError(
            "Can't calculate percentages for an undefined distribution."
        )

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        return tile(params["loc"].reshape((params.shape[0], 1)), percentages.shape[1])


class NoUncertainty(UndefinedUncertainty):
    """No uncertainty: every sample is ``loc``, which is not validated, as for
    :class:`UndefinedUncertainty`."""

    id = 1
    description = "No uncertainty"
