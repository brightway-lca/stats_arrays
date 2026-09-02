from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError
from stats_arrays.utils import ParamsArray, one_row_params_array


class DiscreteUniform(UncertaintyBase):
    """
    The discrete uniform distribution includes all integer values from the ``minimum`` up to, but excluding the ``maximum``.

    See https://en.wikipedia.org/wiki/Uniform_distribution_(discrete).
    """

    id = 7
    description = "Discrete uniform uncertainty"

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        bad = np.isnan(params["maximum"])
        if bad.any():
            raise InvalidParamsError(
                f"Maximum values must always be defined. {cls._fmt_bad_rows(bad)}"
            )
        bad = params["minimum"] >= params["maximum"]
        if bad.any():
            raise ImproperBoundsError(
                f"minimum >= maximum. {cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def fix_nan_minimum(cls, params: ParamsArray) -> ParamsArray:
        mask = np.isnan(params["minimum"])
        params["minimum"][mask] = 0
        return params

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
        params = cls.fix_nan_minimum(params)
        # randint has different behaviour than e.g. uniform. We can't pass in
        # arrays, but have to process them line by line.
        return np.vstack(
            [
                seeded_random.randint(
                    params["minimum"][i],  # Minimum (low)
                    params["maximum"][i],  # Maximum (high)
                    size=size,
                )
                for i in range(params.shape[0])
            ]
        )

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        params = cls.fix_nan_minimum(params)
        for row in range(params.shape[0]):
            results[row, :] = stats.randint.cdf(
                vector[row, :],
                params[row]["minimum"],  # low (shape param)
                params[row]["maximum"],  # high (shape param)
            )
        return results

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        params = cls.fix_nan_minimum(params)
        scale = (params["maximum"] - params["minimum"]).reshape(params.shape[0], 1)
        return percentages * scale + params["minimum"].reshape(params.shape[0], 1)

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> dict:
        params = cls.fix_nan_minimum(params)
        minimum = float(params["minimum"].flat[0])
        maximum = float(params["maximum"].flat[0])
        # The support is {minimum, ..., maximum - 1}, i.e. scipy's `randint(low, high)`
        # with `high` excluded. For the integers a..b inclusive, mean and median are
        # both (a + b) / 2, see
        # https://en.wikipedia.org/wiki/Discrete_uniform_distribution. With
        # b = maximum - 1 that is (minimum + maximum - 1) / 2. For an even count the
        # median is the midpoint of the two middle values, as `numpy.median` defines it.
        centre = (minimum + maximum - 1) / 2
        return {
            "mean": centre,
            "mode": None,
            "median": centre,
            "lower": minimum,
            "upper": maximum,
        }

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ParamsArray, xs: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        params = cls.fix_nan_minimum(params)
        if xs is None:
            xs = np.array([params["minimum"], params["maximum"]])
        percentage = 1 / (params["maximum"] - params["minimum"])
        ys = np.array([float(percentage) for x in xs])
        return np.array([float(x) for x in xs]), ys
