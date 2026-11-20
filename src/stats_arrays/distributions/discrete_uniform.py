from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.errors import ImproperBoundsError, InvalidParamsError
from stats_arrays.utils import (
    ParamsArray,
    RandomSource,
    StatisticsResult,
    one_row_params_array,
)

#: Largest magnitude of a ``DiscreteUniform`` bound: float64 represents every
#: integer from ``-2**53`` to ``2**53`` exactly, but not ``2**53 + 1``.
MAX_EXACT_INTEGER = 2.0**53


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
        # The support must be finite. Infinity would pass the integer check
        # below, as ``inf == floor(inf)``. Checked before the order of the
        # bounds, so that every infinite bound gives this error.
        bad = np.isinf(params["maximum"]) | np.isinf(params["minimum"])
        if bad.any():
            raise InvalidParamsError(
                f"Minimum and maximum values must be finite. {cls._fmt_bad_rows(bad)}"
            )
        # Samples are float64, which holds every integer exactly only up to
        # 2**53 in magnitude. Beyond that, draws are rounded, and can land on
        # the excluded maximum. This limit also keeps the cast of the bounds to
        # int64 in ``random_variables`` defined. A NaN minimum (meaning 0) passes.
        bad = (np.abs(params["minimum"]) > MAX_EXACT_INTEGER) | (
            np.abs(params["maximum"]) > MAX_EXACT_INTEGER
        )
        if bad.any():
            raise InvalidParamsError(
                "Minimum and maximum values must be between -2**53 and 2**53, "
                "where float64 represents every integer exactly. "
                f"{cls._fmt_bad_rows(bad)}"
            )
        # A NaN minimum means 0, so compare with that: ``NaN >= maximum`` is
        # always false, and would let an empty support through.
        bad = cls.fix_nan_minimum(params)["minimum"] >= params["maximum"]
        if bad.any():
            # Only explain the default if it is why a row failed
            hint = (
                " (a NaN minimum is read as 0)"
                if np.isnan(params["minimum"][bad]).any()
                else ""
            )
            raise ImproperBoundsError(
                f"minimum >= maximum{hint}. {cls._fmt_bad_rows(bad)}"
            )
        # The support is a range of integers, and `cdf` and `ppf` return NaN for
        # fractional bounds. A NaN minimum is allowed, and defaults to zero.
        bad = (params["maximum"] != np.floor(params["maximum"])) | (
            ~np.isnan(params["minimum"])
            & (params["minimum"] != np.floor(params["minimum"]))
        )
        if bad.any():
            raise InvalidParamsError(
                f"Minimum and maximum values must be integers. {cls._fmt_bad_rows(bad)}"
            )

    @classmethod
    def fix_nan_minimum(cls, params: ParamsArray) -> ParamsArray:
        """Return ``params`` with a NaN ``minimum`` replaced by 0.

        Returns a copy when there is a NaN to replace, and ``params`` itself
        otherwise; the array passed in is never changed.
        """
        mask = np.isnan(params["minimum"])
        if mask.any():
            params = params.copy()
            params["minimum"][mask] = 0
        return params

    @classmethod
    def random_variables(
        cls,
        params: ParamsArray,
        size: int,
        seeded_random: Optional[RandomSource] = None,
    ) -> npt.NDArray:
        if seeded_random is None:
            seeded_random = np.random.default_rng()
        params = cls.fix_nan_minimum(params)
        if isinstance(seeded_random, np.random.Generator):
            # ``Generator.integers`` broadcasts array bounds, and, like
            # ``randint``, excludes ``high``.
            return seeded_random.integers(
                params["minimum"].astype(np.int64),
                params["maximum"].astype(np.int64),
                size=(size, params.shape[0]),
            ).T
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
        from scipy import stats

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
        from scipy import stats

        percentages = cls.check_2d_inputs(params, percentages)
        params = cls.fix_nan_minimum(params)
        return stats.randint.ppf(
            percentages,
            params["minimum"].reshape(-1, 1),
            params["maximum"].reshape(-1, 1),
        )

    @classmethod
    @one_row_params_array
    def statistics(cls, params: ParamsArray) -> StatisticsResult:
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
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Return the probability mass function.

        ``xs`` defaults to the support, ``minimum, ..., maximum - 1``. Each
        value in the support has probability ``1 / (maximum - minimum)``, and
        any other value, including non-integers, has probability 0.
        """
        from scipy import stats

        params = cls.fix_nan_minimum(params)
        minimum = float(params["minimum"].flat[0])
        maximum = float(params["maximum"].flat[0])
        if xs is None:
            xs = np.arange(minimum, maximum)
        xs = np.asarray(xs, dtype=float).reshape(-1)
        return xs, stats.randint.pmf(xs, minimum, maximum)
