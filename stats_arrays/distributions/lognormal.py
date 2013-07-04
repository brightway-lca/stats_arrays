from __future__ import division
from ..errors import InvalidParamsError, ImproperBoundsError
from ..utils import one_row_params_array
from .base import UncertaintyBase
from scipy import stats
import numpy as np


class LognormalUncertainty(UncertaintyBase):
    id = 2
    description = "Lognormal uncertainty"

    @classmethod
    def validate(cls, params):
        """Custom validation because mean gets log-transformed"""
        if np.isnan(params['loc']).sum():
            raise InvalidParamsError(
                "Real location (mu) values are required for"
                " lognormal uncertainties."
            )
        if np.isnan(params['scale']).sum() or (params['scale'] <= 0).sum():
            raise InvalidParamsError(
                "Real, positive scale (sigma) values are required for"
                " lognormal uncertainties."
            )

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        if not seeded_random:
            seeded_random = np.random
        data = seeded_random.lognormal(
            params['loc'],  # Mu
            params['scale'],  # Sigma
            size=(size, params.shape[0])
        ).T
        # Negative is needed because log loses sign information.
        # Error handling not included, as this loop is called many times
        data[params['negative'], :] = -1 * data[params['negative'], :]
        return data

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        vector[params['negative']] = -1 * vector[params['negative']]
        results = np.zeros(vector.shape)
        for row in range(params.shape[0]):
            results[row, :] = stats.lognorm.cdf(
                vector[row, :],
                params['scale'][row],
                scale=np.exp(params['loc'][row])
            )
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = np.zeros(percentages.shape)
        for row in range(percentages.shape[0]):
            results[row, :] = stats.lognorm.ppf(
                percentages[row, :],
                params['scale'][row],
                scale=np.exp(params['loc'][row])
            )
        results[params['negative']] = -1 * results[params['negative']]
        return results

    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        negative = -1 if bool(params['negative']) else 1
        geometric_mu = float(np.exp(params['loc']))
        sigma = float(params['scale'])
        mu = float(params['loc'])
        geometric_sigma = float(np.exp(sigma))
        mean = np.exp(mu + (sigma ** 2) / 2)
        mode = np.exp(mu - sigma ** 2)
        ci_95_lower = geometric_mu / (geometric_sigma ** 2)
        ci_95_upper = geometric_mu * (geometric_sigma ** 2)
        if negative == -1:
            ci_95_lower, ci_95_upper = ci_95_upper, ci_95_lower
        return {
            'median': negative * geometric_mu,
            'mode': negative * mode,
            'mean': negative * mean,
            'lower': negative * ci_95_lower,
            'upper': negative * ci_95_upper
        }

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        """Generate probability distribution function for lognormal distribution."""
        if xs is None:
            # Find nice range to graph
            if np.isnan(params['minimum']):
                minimum = params['scale'] / (np.exp(params['shape']) **
                    cls.standard_deviations_in_default_range)
            else:
                minimum = np.abs(params['minimum'])
            if np.isnan(params['maximum']):
                maximum = params['scale'] * (np.exp(params['shape']) **
                    cls.standard_deviations_in_default_range)
            else:
                maximum = np.abs(params['minimum'])

            xs = np.linspace(
                minimum,
                maximum,
                cls.default_number_points_in_pdf
            ).ravel()
            print xs.shape

        if params['negative']:
            xs = -1 * xs
        ys = stats.lognorm.pdf(xs, params['scale'], scale=np.exp(params['loc']))
        if params['negative']:
            xs = -1 * xs
        return xs, ys.ravel()
