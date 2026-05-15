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
        bad = (params["loc"] < 0) | (params["loc"] > 1)
        if bad.any():
            raise InvalidParamsError(
                f"Bernoulli uncertainty requires loc values between 0 and 1 (inclusive). "
                f"{cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        if seeded_random is None:
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
        p = params["loc"].reshape(-1, 1)
        return np.where(vector < 0, 0.0, np.where(vector < 1, 1 - p, 1.0))

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        return (percentages > 1 - params["loc"].reshape(-1, 1)) * 1.0
