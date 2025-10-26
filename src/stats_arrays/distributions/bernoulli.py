from typing import Optional

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import BoundedUncertaintyBase
from stats_arrays.utils import ParamsArray


class BernoulliUncertainty(BoundedUncertaintyBase):
    id = 6
    description = "Bernoulli uncertainty"

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        data = np.zeros((params.shape[0], size))
        if not seeded_random:
            seeded_random = np.random.RandomState()
        adjusted_means, scale = cls.rescale(params)
        adjusted_means.resize(params.shape[0], 1)
        # nums = seeded_random.random_sample(size * params.shape[0])
        # nums_rs = nums.reshape(params.shape[0], size)
        mask = (
            seeded_random.random_sample(size * params.shape[0]).reshape(
                (params.shape[0], size)
            )
            >= adjusted_means
        )
        data[mask] = 1
        return data

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        # Turn into column vector
        mean = np.array(params["loc"])
        mean.resize(params.shape[0], 1)
        return (vector >= mean) * 1

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        adjusted_means, scale = cls.rescale(params)
        length = params.shape[0]
        return (percentages >= adjusted_means.reshape(length, 1)) * scale.reshape(
            length, 1
        ) + params["minimum"].reshape(length, 1)
