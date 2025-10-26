from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import InvalidParamsError
from stats_arrays.utils import ParamsArray, one_row_params_array


class NormalUncertainty(UncertaintyBase):
    id = 3
    description = "Normal uncertainty"

    @classmethod
    def validate(cls, params: ParamsArray) -> None:
        if np.isnan(params["scale"]).sum() or (params["scale"] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive scale (sigma) values are required"
                " for normal uncertainties."
            )
        if np.isnan(params["loc"]).sum():
            raise InvalidParamsError(
                "Real loc (mu) values are required for normal uncertainties."
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
        return seeded_random.normal(
            params["loc"], params["scale"], size=(size, params.shape[0])
        ).T

    @classmethod
    def cdf(cls, params: ParamsArray, vector: npt.NDArray) -> npt.NDArray:
        vector = cls.check_2d_inputs(params, vector)
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.norm.cdf(
                vector[row, :], loc=params["loc"][row], scale=params["scale"][row]
            )
        return results

    @classmethod
    def ppf(cls, params: ParamsArray, percentages: npt.NDArray) -> npt.NDArray:
        percentages = cls.check_2d_inputs(params, percentages)
        results = np.zeros(percentages.shape)
        for row in range(percentages.shape[0]):
            results[row, :] = stats.norm.ppf(
                percentages[row, :], loc=params["loc"][row], scale=params["scale"][row]
            )
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> dict:
        return {
            "mean": float(params["loc"].flat[0]),
            "mode": float(params["loc"].flat[0]),
            "median": float(params["loc"].flat[0]),
            "lower": float((params["loc"] - 2 * params["scale"]).flat[0]),
            "upper": float((params["loc"] + 2 * params["scale"]).flat[0]),
        }

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ParamsArray, xs: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if xs is None:
            if np.isnan(params["minimum"]):
                lower = (
                    params["loc"]
                    - params["scale"] * cls.standard_deviations_in_default_range
                )
            else:
                lower = params["minimum"]
            if np.isnan(params["maximum"]):
                upper = (
                    params["loc"]
                    + params["scale"] * cls.standard_deviations_in_default_range
                )
            else:
                upper = params["maximum"]
            xs = np.arange(
                lower.flat[0],
                upper.flat[0],
                (upper - lower).flat[0] / cls.default_number_points_in_pdf,
            )

        ys = stats.norm.pdf(xs, params["loc"], params["scale"])
        return xs, ys.reshape(ys.shape[1])
