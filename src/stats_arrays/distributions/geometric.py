from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from stats_arrays.distributions.base import BoundedUncertaintyBase
from stats_arrays.errors import ImproperBoundsError
from stats_arrays.utils import ParamsArray, one_row_params_array


class UniformUncertainty(BoundedUncertaintyBase):
    """Continuous uniform distribution. In SciPy, the uniform distribution is defined from loc to loc+scale."""

    id = 4
    description = "Uniform uncertainty"

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
        return seeded_random.uniform(
            params["minimum"],  # Minimum (low)
            params["maximum"],  # Maximum (high)
            size=(size, params.shape[0]),
        ).T

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.uniform.cdf(
                vector[row, :],
                loc=params[row]["minimum"],
                scale=params[row]["maximum"] - params[row]["minimum"],
            )
        return results

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        scale = (params["maximum"] - params["minimum"]).reshape(params.shape[0], 1)
        return percentages * scale + params["minimum"].reshape(params.shape[0], 1)

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> dict:
        mean = (params["maximum"] + params["minimum"]) / 2
        return {
            "mean": mean,
            "mode": mean,
            "median": mean,
            "lower": params["minimum"],
            "upper": params["maximum"],
        }

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ParamsArray, xs: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if xs is None:
            xs_array = np.array([params["minimum"], params["maximum"]]).reshape(
                2,
            )
        else:
            xs_array = xs
        percentage = float((1 / (params["maximum"] - params["minimum"])).flat[0])
        ys = np.array([percentage for x in xs_array])
        return xs_array, ys


class TriangularUncertainty(BoundedUncertaintyBase):
    id = 5
    description = "Triangular uncertainty"

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        super(TriangularUncertainty, cls).validate(params)
        if (params["loc"] > params["maximum"]).sum() or (
            params["loc"] < params["minimum"]
        ).sum():
            raise ImproperBoundsError("Most likely value outside the given bounds.")

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[np.random.RandomState] = None,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.RandomState()
        return seeded_random.triangular(
            params["minimum"],  # Left
            params["loc"],  # Mode
            params["maximum"],  # Right
            size=(size, params.shape[0]),
        ).T

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        # Adjust parameters to (0,1) range
        adjusted_means, scale = cls.rescale_to_unitary_interval(params)
        # To be broadcasted correctly, scale and mins must be column vectors
        scale.resize(scale.shape[0], 1)
        mins = np.array(params["minimum"])
        mins.resize(params.shape[0], 1)
        # Adjust values to use cdf for to (0,1) range
        adjusted_vector = (vector - mins) / scale
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.triang.cdf(
                adjusted_vector[row, :], adjusted_means[row]
            )
        return results

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        adjusted_means, scale = cls.rescale_to_unitary_interval(params)
        scale.resize(scale.shape[0], 1)
        adjusted_means.resize(scale.shape[0], 1)
        return cls.rescale_vector_to_params(
            params, stats.triang.ppf(percentages, adjusted_means)
        )

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> dict:
        mode = float(params["loc"].flat[0])
        mean = float(((params["minimum"] + params["maximum"] + mode) / 3).flat[0])
        lower, median, upper = cls.ppf(
            params, np.array([[0.0125, 0.5, 0.9875]])
        ).ravel()
        return {
            "mean": mean,
            "median": float(median.flat[0]),
            "mode": mode,
            "lower": float(lower.flat[0]),
            "upper": float(upper.flat[0]),
        }

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ParamsArray, xs: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if xs is None:
            lower = params["minimum"]
            upper = params["maximum"]
            mode = params["loc"]
            if not mode:
                mode = (upper + lower) / 2
            xs = np.array([float(x.flat[0]) for x in (lower, mode, upper)])
            ys = np.array([0, float(((mode - lower) / (upper - lower)).flat[0]), 0])
        else:
            adjusted_means, scale = cls.rescale_to_unitary_interval(params)
            adj_xs = (xs - params["minimum"]) / scale
            ys_0_1_interval = stats.triang.pdf(adj_xs, adjusted_means)
            ys = ys_0_1_interval / scale
        return xs, ys
