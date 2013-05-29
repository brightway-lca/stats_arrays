# -*- coding: utf-8 -*
from __future__ import division
from .errors import UnknownUncertaintyType
from .uncertainty_choices import *
from numpy import array, zeros, random, hstack, arange, isnan, tile, argsort


class RandomNumberGenerator(object):
    def __init__(self, uncertainty_type, params, size=1,
            convert_lognormal=False, maximum_iterations=100, seed=None,
            **kwargs):
        """params is a structured array, with three different datatypes. The columns, columns labels, and datatypes are:
            input: Input id. Unsigned int.
            output: Output id. Unsigned int.
            mean: Mean, mode for triangular, can be geometric mean for lognormal is convert_lognormal is True. Float.
            negative: Mean is negative. Needed for neagtive lognormals. Boolean.
            scale: Sigma. Default is NaN. Float.
            minimum: Miniumum. Default is NaN. Float.
            maximum: Maximum. Default is NaN. Float.

        The call is create params is:
            params = zeros((size,), dtype=[('mean', 'f4'), ('negative', 'b1'), ('scale', 'f4'), ('minimum', 'f4'), ('maximum', 'f4')])
            params['maximum'] = params['minimum'] = params['scale'] = NaN"""
        self.params = params
        self.length = self.params.shape[0]
        self.size = size
        self.uncertainty_type = uncertainty_type
        self.maximum_iterations = maximum_iterations
        # Needed even if seed=None because of celery & multiprocessing issues
        self.random = random.RandomState(seed)
        self.verify_uncertainty_type()
        self.verify_params()
        if convert_lognormal:
            self.convert_lognormal_values()

    def verify_params(self, params=None, uncertainty_type=None):
        """Verify that parameters are within bounds. Mean is not restricted to bounds, unless the distribution requires it (e.g. triangular)."""
        if params is None:  # Can't convert array to boolean
            params = self.params
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        uncertainty_type.validate(params)

    def verify_uncertainty_type(self, uncertainty_type=None):
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if uncertainty_type not in UncertaintyChoices():
            raise UnknownUncertaintyType

    def convert_lognormal_values(self, params=None):
        if params is None:
            params = self.params
        if self.uncertainty_type == LognormalUncertainty:
            LognormalUncertainty.set_negative_flag(params)

    def generate_random_numbers(self, uncertainty_type=None, params=None,
            size=None):
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if params == None:  # Can't convert array to boolean
            params = self.params
        if not size:
            size = self.size
        return uncertainty_type.bounded_random_variables(params, size,
            self.random, self.maximum_iterations)

    def go(self):
        # Shortcut
        return self.generate_random_numbers()


class MCRandomNumberGenerator(RandomNumberGenerator):
    """
A random number generator that understands the exchange array produced by the LCA class.

The generation of numbers for individual distributions is to left to the distributions themselves. This class expects a structured array with the standard uncertainty columns (loc, scale, etc.) and a integer column of uncertainty_type ids.
    """
    def __init__(self, params, maximum_iterations=50, seed=None, sort=True,
            **kwargs):
        self.params = params.copy()
        self.length = self.params.shape[0]
        # Only one value form each distribution instance
        self.size = 1
        self.maximum_iterations = maximum_iterations
        self.choices = UncertaintyChoices()
        self.random = random.RandomState(seed)
        self.verify_params()
        self.convert_lognormal_values(self.params)

        self.sorted = not sort
        if not self.sorted:
            self.ordering = argsort(self.params["uncertainty_type"])
            self.params = self.params[self.ordering]

        self.positions = self.get_positions()

    def get_positions(self):
        """Construct dictionary of where each distribution stops/starts"""
        d = {}
        for choice in self.choices:
            d[choice] = (self.params['uncertainty_type'] == choice.id).sum()
        return d

    def verify_params(self):
        """Verify parameters using distribution class methods"""
        for uncertainty_type in self.choices:
            mask = self.params['uncertainty_type'] == uncertainty_type.id
            if mask.sum():
                uncertainty_type.validate(self.params[mask])

    def convert_lognormal_values(self, params=None):
        if params is None:
            params = self.params
        lognormal_mask = params['uncertainty_type'] == \
            LognormalUncertainty.id
        LognormalUncertainty.set_negative_flag(
            params[lognormal_mask])
        return params

    def next(self):
        if not hasattr(self, "random_data"):
            self.random_data = zeros(self.length)

        offset = 0
        for uncertainty_type in self.choices:
            size = self.positions[uncertainty_type]
            if not size:
                continue
            random_data = uncertainty_type.bounded_random_variables(
                self.params[offset:size + offset], self.size, self.random,
                self.maximum_iterations)
            if len(random_data.shape) == 2:
                random_data = random_data[:, 0]  # Restore to 1-d
            self.random_data[offset:size + offset] = random_data
            offset += size

        if not self.sorted:
            self.random_data = self.random_data[argsort(self.ordering)]

        return self.random_data


class LatinHypercubeRNG(MCRandomNumberGenerator):
    """
A random number generator that pre-calculates a sample space to draw from.

.. rubric:: Inputs

* params : A :ref:`params-array` which gives parameters for distributions (one distribution per row).
* seed : An integer (or array of integers) to seed the `NumPy random number generator <http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.mtrand.RandomState.html#numpy.random.mtrand.RandomState>`_.
* samples : An integer number of samples to construct for each distribution.
    """
    def __init__(self, params, seed=None, samples=10, **kwargs):
        self.params = params
        self.length = self.params.shape[0]
        self.row_index = arange(self.length)
        self.size = 1
        self.samples = samples
        self.choices = UncertaintyChoices()

        if seed:
            self.random = random.RandomState(seed)
        else:
            self.random = random

        self.uncertainty_type = None
        self.convert_lognormal = False
        self.convert_lognormal_values()

        self.verify_params()
        self.build_hypercube()

    def build_hypercube(self):
        """Build an array, of shape `self.length` rows by `self.samples` columns, which contains the sample space to be drawn from when doing Latin Hypercubic sampling.

Each row represents a different data point and distribution. The final sample space is `self.hypercube`. All distributions from `uncertainty_choices` are usable, and bounded distributions are also fine.

.. rubric:: Builds

self.hypercube : Numpy array with dimensions `self.length` by `self.samples`."""
        step_size = 1 / (self.samples + 1)
        # Define the beginning points and step sizes - not all the same because
        # some distributions are bounded. Make adjustments to generic values
        # later.
        inputs = tile(array((step_size, step_size)), self.length).reshape(
            self.length, 2)
        for uncertainty_type in self.choices:
            mask = self.params['uncertainty_type'] == uncertainty_type.id
            if not mask.sum():
                continue
            subarray = self.params[mask]
            # Adjust inputs when bounds are present. Easiest to do in three
            # discrete steps. First, when only a lower bound is present.
            only_min_mask = ~isnan(subarray['minimum']) * \
                isnan(subarray['maximum'])
            if only_min_mask.sum():
                mins = uncertainty_type.cdf(subarray[only_min_mask],
                    subarray[only_min_mask]['minimum'])
                steps = (1 - mins) / (self.samples + 1)
                inputs[mask, :][only_min_mask] = hstack((mins + steps,
                    steps))
            # Next, if only a max bound is present
            only_max_mask = isnan(subarray['minimum']) * \
                ~isnan(subarray['maximum'])
            if only_max_mask.sum():
                maxs = uncertainty_type.cdf(subarray[only_max_mask],
                    subarray[only_max_mask]['maximum'])
                steps = maxs / (self.samples + 1)
                inputs[mask, :][only_max_mask] = hstack((steps, steps))
            # Finally, if both min and max bounds are present
            both_mask = ~isnan(subarray['minimum']) * \
                ~isnan(subarray['maximum'])
            if both_mask.sum():
                mins = uncertainty_type.cdf(subarray[both_mask],
                    subarray[both_mask]['minimum'])
                maxs = uncertainty_type.cdf(subarray[both_mask],
                    subarray[both_mask]['maximum'])
                steps = (maxs - mins) / (self.samples + 1)
                inputs[mask, :][both_mask] = hstack((mins + steps,
                    steps))
        # Percentages is now a list, samples wide, of the percentages covered
        # by the bounded or unbounded distributions.
        self.percentages = inputs[:, 0].reshape(self.length, 1) + arange(0,
            self.samples, 1) * inputs[:, 1].reshape(self.length, 1)
        # Transform percentages into a sample space
        self.hypercube = zeros((self.length, self.samples))
        for uncertainty_type in self.choices:
            mask = self.params['uncertainty_type'] == uncertainty_type.id
            if not mask.sum():
                continue
            self.hypercube[mask, :] = \
                uncertainty_type.ppf(params=self.params[mask],
                percentages=self.percentages[mask, :])
        self.hypercube[self.params['negative'], :] = \
            -1 * self.hypercube[self.params['negative'], :]

    def iterate(self):
        """Draw directly from pre-computed sample space."""
        return self.hypercube[self.row_index,
            self.random.randint(self.samples, size=self.length)]
