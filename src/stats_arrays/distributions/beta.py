from __future__ import division

from numpy import isnan, linspace, random, zeros, newaxis
from scipy import stats

from stats_arrays.errors import InvalidParamsError, ImproperBoundsError
from stats_arrays.utils import one_row_params_array
from stats_arrays.distributions.base import UncertaintyBase


class BetaUncertainty(UncertaintyBase):
    """
    The Beta distribution has the probability distribution function:

    .. math:: f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}(1 - x)^{\\beta - 1},

    where the normalisation, *B*, is the beta function:

    .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}(1 - t)^{\\beta - 1} dt

    The :math:`\\alpha` parameter is ``loc``, and :math:`\\beta` is ``shape``. By default, the Beta distribution is defined from 0 to 1; the lower and upper bounds can be rescaled with the ``minimum`` and ``maximum`` parameters.

    Wikipedia: `Beta distribution <http://en.wikipedia.org/wiki/Beta_distribution>`_
    """

    id = 10
    description = "Beta uncertainty"

    @classmethod
    def validate(cls, params):
        if (params["loc"] > 0).sum() != params.shape[0]:
            raise InvalidParamsError(
                "Real, positive alpha values are" + " required for Beta uncertainties."
            )
        if (params["shape"] > 0).sum() != params.shape[0]:
            raise InvalidParamsError(
                "Real, positive beta values are" + " required for Beta uncertainties."
            )
        if (params["minimum"] >= params["maximum"]).sum() or (
            params["maximum"] <= params["minimum"]
        ).sum():
            raise ImproperBoundsError("Min/max inconsistency.")

    @classmethod
    def _rescale(cls, params, results):
        """Rescale results from [0, 1] to [minimum, maximum] range.

        The correct scaling formula is: results = results * scale + minimum
        where scale = maximum - minimum.
        """
        minimum, scale = cls._minimum_scale(params)

        # Handle broadcasting for multiple rows
        if results.ndim == 2 and minimum.ndim == 1:
            # results shape: (n_rows, n_samples), minimum/scale shape: (n_rows,)
            results = results * scale[:, newaxis] + minimum[:, newaxis]
        else:
            # Single row case or matching dimensions
            results = results * scale + minimum

        return results

    @classmethod
    def _minimum_scale(cls, params):
        minimum = params["minimum"].copy()
        minimum[isnan(minimum)] = 0
        scale = params["maximum"].copy()
        scale[isnan(scale)] = 1
        scale -= minimum
        return minimum, scale

    @classmethod
    def random_variables(cls, params, size, seeded_random=None, transform=False):
        if not seeded_random:
            seeded_random = random
        return cls._rescale(
            params,
            seeded_random.beta(
                params["loc"], params["shape"], size=(size, params.shape[0])
            ).T,
        )

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        loc, scale = cls._minimum_scale(params)
        for row in range(params.shape[0]):
            results[row, :] = stats.beta.cdf(
                vector[row, :],
                params["loc"][row],
                params["shape"][row],
                loc=loc[row],
                scale=scale[row],
            )
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = zeros(percentages.shape)
        loc, scale = cls._minimum_scale(params)
        for row in range(percentages.shape[0]):
            results[row, :] = stats.beta.ppf(
                percentages[row, :],
                params["loc"][row],
                params["shape"][row],
                loc=loc[row],
                scale=scale[row],
            )
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        alpha = float(params["loc"][0][0])
        beta = float(params["shape"][0][0])

        minimum, scale = cls._minimum_scale(params)
        minimum = float(minimum[0][0])
        scale = float(scale[0][0])

        if alpha <= 1 or beta <= 1:
            mode = "Undefined"
        else:
            mode = ((alpha - 1) / (alpha + beta - 2)) * scale + minimum
        return {
            "mean": (alpha / (alpha + beta)) * scale + minimum,
            "mode": mode,
            "median": "Not Implemented",
            "lower": "Not Implemented",
            "upper": "Not Implemented",
        }

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        loc, scale = cls._minimum_scale(params)
        loc = float(loc[0][0])
        scale = float(scale[0][0])

        if xs is None:
            xs = linspace(loc, loc + scale, cls.default_number_points_in_pdf)
        ys = stats.beta.pdf(xs, params["loc"], params["shape"], loc=loc, scale=scale)
        return xs, ys.reshape(ys.shape[1])
