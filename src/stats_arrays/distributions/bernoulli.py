from typing import Optional

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray


class BernoulliUncertainty(UncertaintyBase):
    id = 6
    description = "Bernoulli uncertainty"

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        """Validate that loc is between 0 and 1 (inclusive)."""
        if (params["loc"] < 0).sum() or (params["loc"] > 1).sum():
            raise InvalidParamsError(
                "Bernoulli uncertainty requires loc values between 0 and 1 (inclusive)."
            )

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        if not seeded_random:
            seeded_random = np.random.RandomState()
        data = np.zeros((params.shape[0], size))
        mask = (
            seeded_random.random_sample(size * params.shape[0]).reshape(
                (params.shape[0], size)
            )
            <= params["loc"]
        )
        data[mask] = 1
        return data

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        return (vector <= params["loc"].reshape(-1, 1)) * 1.0

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        return (percentages <= params["loc"].reshape(-1, 1)) * 1.0
