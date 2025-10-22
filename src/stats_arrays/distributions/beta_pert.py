from typing import Optional
from numpy import isnan, ndarray, nan, random

from stats_arrays.distributions.beta import BetaUncertainty
from stats_arrays.errors import InvalidParamsError, ImproperBoundsError
from stats_arrays.utils import one_row_params_array


class BetaPERTUncertainty(BetaUncertainty):
    """
    The Beta PERT distribution is a particular form of the Beta distribution which takes these input parameters:

    #. ``minimum``: ``A``, the minimum value
    #. ``loc``: ``B``, the mean value
    #. ``maximum``: ``C``, the maximum value
    #. ``shape``: ``:math:`\\lambda```, the optional shape parameter

    The standard default shape parameter is 4, though according to Wikipedia values from 2 to 3.5 are commonly used. A higher value makes the distribution more concentrated on the mean value.

    Wikipedia: `Beta PERT distribution <https://en.wikipedia.org/wiki/PERT_distribution>`_
    """

    id = 13
    description = "Beta PERT uncertainty"

    @classmethod
    def validate(cls, params):
        if isnan(params["minimum"]).sum():
            raise InvalidParamsError(
                "Real, positive `A` values are required for Beta PERT uncertainties."
            )
        if isnan(params["loc"]).sum():
            raise InvalidParamsError(
                "Real, positive `B` values are required for Beta PERT uncertainties."
            )
        if isnan(params["maximum"]).sum():
            raise InvalidParamsError(
                "Real, positive `C` values are required for Beta PERT uncertainties."
            )
        if (params["minimum"] > params["loc"]).sum() or (
            params["loc"] > params["maximum"]
        ).sum():
            raise ImproperBoundsError("`A <= B <= C` not respected.")
        if (params["minimum"] == params["maximum"]).sum():
            raise ImproperBoundsError("`A` and `C` have the same values.")
        # Check lambda values where they are provided (not NaN)
        provided_lambda = params[~isnan(params["scale"])]
        if (provided_lambda["scale"] <= 0).sum():
            raise InvalidParamsError("Lambda values must be greater than zero.")

    @classmethod
    def _as_beta(cls, params: ndarray, default_lambda: float = 4.0) -> ndarray:
        """Calculate α and β values for Beta distribution from PERT A/B/C inputs."""
        beta = params.copy()
        # Use provided lambda values or default
        lambda_values = params["scale"].copy()
        missing_lambda_mask = isnan(lambda_values)
        lambda_values[missing_lambda_mask] = default_lambda

        a, b, c = params["minimum"], params["loc"], params["maximum"]
        # Set Beta distribution parameters
        beta["scale"] = nan
        beta["loc"] = 1 + lambda_values * ((b - a) / (c - a))  # alpha
        beta["shape"] = 1 + lambda_values * ((c - b) / (c - a))  # beta
        # Set minimum and maximum for scaling from [0, 1] to [A, C]
        beta["minimum"] = a
        beta["maximum"] = c
        return beta

    @classmethod
    def random_variables(
        cls,
        params: ndarray,
        size: int,
        seeded_random: Optional[random.RandomState] = None,
        transform: bool = False,
        default_lambda: float = 4.0,
    ):
        return BetaUncertainty.random_variables(
            params=cls._as_beta(params=params, default_lambda=default_lambda),
            size=size,
            seeded_random=seeded_random,
            transform=transform,
        )

    @classmethod
    def cdf(cls, params: ndarray, vector: ndarray, default_lambda: float = 4.0):
        return BetaUncertainty.cdf(
            params=cls._as_beta(params=params, default_lambda=default_lambda),
            vector=vector,
        )

    @classmethod
    def ppf(cls, params: ndarray, percentages: ndarray, default_lambda: float = 4.0):
        return BetaUncertainty.ppf(
            params=cls._as_beta(params=params, default_lambda=default_lambda),
            percentages=percentages,
        )

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ndarray, default_lambda: float = 4.0):
        return BetaUncertainty.statistics(
            params=cls._as_beta(params=params, default_lambda=default_lambda)
        )

    @classmethod
    @one_row_params_array
    def pdf(
        cls, params: ndarray, xs: Optional[ndarray] = None, default_lambda: float = 4.0
    ):
        return BetaUncertainty.pdf(
            params=cls._as_beta(params=params, default_lambda=default_lambda), xs=xs
        )
