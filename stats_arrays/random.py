from .errors import UnknownUncertaintyType
from .uncertainty_choices import *
from collections.abc import Iterable
import numpy as np


class RandomNumberGenerator(Iterable):

    def __init__(self, uncertainty_type, params, size=1,
                 maximum_iterations=100, seed=None,
                 **kwargs):
        """
Create a random number generator from a :ref:`params-array` and an uncertainty distribution.

Upon instantiation, the class checks that:

* The ``minimum`` and ``maximum`` bounds, if any, are reasonable
* The given uncertainty type can be used

``uncertainty_type`` is not required to be a subclass of :ref:`UncertaintyBase`, but needs to have the method ``bounded_random_variables``.

The returned class instance can be called directly::

    >>> from stats_arrays import RandomNumberGenerator, TriangularUncertainty
    >>> params = TriangularUncertainty.from_dicts(
    ...     {'loc': 5, 'minimum': 3, 'maximum': 10},
    ...     {'loc': 1, 'minimum': 0.7, 'maximum': 4.4}
    ...     )
    >>> rng = RandomNumberGenerator(TriangularUncertainty, params)
    >>> rng.generate_random_numbers()
    array([[ 8.00843856],
       [ 1.54968237]])

but can also be used as an iterator::

    >>> zip(range(2), rng)
    [(0, array([[ 5.34298156],
       [ 1.02447677]])),
     (1, array([[ 5.45360508],
       [ 1.99372889]]))]

Args:
    * **uncertainty_type** (object): An uncertainty type class (subclass of ``stats_arrays.distributions.UncertaintyBase``)
    * **params** (array): The :ref:`params-array`
    * *size* (int, optional): The number of samples to draw from each parameter. Default is ``1``.
    * *maximum_iterations* (int, optional): The number of times to draw samples that fit within the given bounds, if any, before raising ``stats_arrays.MaximumIterationsError``. Default is ``100``.
    * *seed* (int, optional): Seed value for the random number generator. Default is ``None``.

Returns:
    A class instance

        """
        self.params = params
        self.length = self.params.shape[0]
        self.size = size
        self.uncertainty_type = uncertainty_type
        self.maximum_iterations = maximum_iterations
        self.random = np.random.RandomState(seed)
        self.verify_uncertainty_type()
        self.verify_params()

    def verify_params(self, params=None, uncertainty_type=None):
        """Verify that parameters are within bounds. Mean is not restricted to bounds, unless the distribution requires it (e.g. triangular)."""
        if params is None:  # Can't convert array to boolean
            params = self.params
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        uncertainty_type.validate(params)

    def verify_uncertainty_type(self, uncertainty_type=None):
        """Make sure the given uncertainty type provides the method ``bounded_random_variables``."""
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if not hasattr(uncertainty_type, u"bounded_random_variables"):
            raise UnknownUncertaintyType(
                u"The provided uncertainty type must have the "
                u"`bounded_random_variables` method."
            )

    def generate_random_numbers(self, uncertainty_type=None, params=None,
                                size=None):
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if params is None:  # Can't convert array to boolean
            params = self.params
        if not size:
            size = self.size
        return uncertainty_type.bounded_random_variables(
            params,
            size,
            self.random,
            self.maximum_iterations
        )

    def next(self):
        return self.generate_random_numbers()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class MCRandomNumberGenerator(Iterable):

    """
A Monte Carlo random number generator that operates on a :ref:`hpa`.

Upon instantiation, the class checks that:

* Each unique ``uncertainty_type`` is a valid choice in ``uncertainty_choices``
* That the parameter array for each uncertainty type validates

The returned class instance can be called directly with ``next``, or can be used as an iterator::

    >>> from stats_arrays import MCRandomNumberGenerator, UncertaintyBase
    >>> params = UncertaintyBase.from_dicts(
    ...     {'loc': 5, 'minimum': 3, 'maximum': 10, 'uncertainty_type': 5},
    ...     {'loc': 1, 'scale': 0.7, 'uncertainty_type': 3}
    ...     )
    >>> mcrng = MCRandomNumberGenerator(params)
    >>> zip(range(2), mcrng)
    [(0, array([ 1.35034874,  5.2705415 ])),
     (1, array([ 5.2705415 ,  1.35034874]))]

Args:
    * **params** (array): The :ref:`hpa`
    * *maximum_iterations* (int, optional): The number of times to draw samples that fit within the given bounds, if any, before raising ``stats_arrays.MaximumIterationsError``. Default is ``100``.
    * *seed* (int, optional): Seed value for the random number generator. Default is ``None``.

Returns:
    A class instance

    """

    def __init__(self, params, maximum_iterations=50, seed=None, **kwargs):
        self.params = params.copy()
        self.length = self.params.shape[0]
        self.maximum_iterations = maximum_iterations
        self.choices = uncertainty_choices
        self.random = np.random.RandomState(seed)
        self.verify_params()
        self.ordering = np.argsort(self.params["uncertainty_type"])
        self.params = self.params[self.ordering]
        self.positions = self.get_positions()

    def get_positions(self):
        """Construct dictionary of where each distribution starts and stops in the sorted parameter array"""
        return dict([(choice, (self.params[u'uncertainty_type'] == choice.id).sum()
                      ) for choice in self.choices])

    def verify_params(self):
        """Verify that all uncertainty types are allowed, and parameter validate using distribution class methods"""
        ids = set(np.unique(self.params[u'uncertainty_type']))
        extra_ids = ids.difference(set([x.id for x in self.choices]))
        if extra_ids:
            raise ValueError(
                u"Uncertainty type id(s) {} are not valid".format(extra_ids)
            )

        for uncertainty_type in self.choices:
            mask = self.params[u'uncertainty_type'] == uncertainty_type.id
            if mask.sum():
                uncertainty_type.validate(self.params[mask])

    def generate(self, samples=1):
        """Generate random samples.

        If ``samples`` is one, return a one-dimensional array. Otherwise returns a ``num_parameters, samples`` array."""
        if samples == 1:
            self.random_data = np.zeros(self.length)
        else:
            self.random_data = np.zeros((self.length, samples))

        offset = 0
        for uncertainty_type in self.choices:
            numparams = self.positions[uncertainty_type]
            if not numparams:
                continue
            random_data = uncertainty_type.bounded_random_variables(
                self.params[offset:numparams + offset],
                samples,
                self.random,
                self.maximum_iterations
            )
            if samples == 1:
                if len(random_data.shape) == 2:
                    random_data = random_data[:, 0]  # Restore to 1-d
                self.random_data[offset:numparams + offset] = random_data
            else:
                self.random_data[offset:numparams + offset, :] = random_data
            offset += numparams

        self.random_data = self.random_data[np.argsort(self.ordering)]
        return self.random_data

    def next(self):
        """Generate a new vector of random numbers"""
        return self.generate()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


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
        self.row_index = np.arange(self.length)
        self.size = 1
        self.samples = samples
        self.choices = uncertainty_choices
        self.random = np.random.RandomState(seed)
        self.uncertainty_type = None

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
        inputs = np.tile(
            np.array((step_size, step_size)),
            self.length
        ).reshape(self.length, 2)
        for uncertainty_type in self.choices:
            mask = self.params['uncertainty_type'] == uncertainty_type.id
            if not mask.sum():
                continue
            subarray = self.params[mask]
            # Adjust inputs when bounds are present. Easiest to do in three
            # discrete steps. First, when only a lower bound is present.
            only_min_mask = ~np.isnan(subarray['minimum']) * \
                np.isnan(subarray['maximum'])
            if only_min_mask.sum():
                mins = uncertainty_type.cdf(subarray[only_min_mask],
                                            subarray[only_min_mask]['minimum'])
                steps = (1 - mins) / (self.samples + 1)
                inputs[mask, :][only_min_mask] = np.hstack((mins + steps,
                                                            steps))
            # Next, if only a max bound is present
            only_max_mask = np.isnan(subarray['minimum']) * \
                ~np.isnan(subarray['maximum'])
            if only_max_mask.sum():
                maxs = uncertainty_type.cdf(subarray[only_max_mask],
                                            subarray[only_max_mask]['maximum'])
                steps = maxs / (self.samples + 1)
                inputs[mask, :][only_max_mask] = np.hstack((steps, steps))
            # Finally, if both min and max bounds are present
            both_mask = ~np.isnan(subarray['minimum']) * \
                ~np.isnan(subarray['maximum'])
            if both_mask.sum():
                mins = uncertainty_type.cdf(subarray[both_mask],
                                            subarray[both_mask]['minimum'])
                maxs = uncertainty_type.cdf(subarray[both_mask],
                                            subarray[both_mask]['maximum'])
                steps = (maxs - mins) / (self.samples + 1)
                inputs[mask, :][both_mask] = np.hstack((mins + steps,
                                                        steps))
        # Percentages is now a list, samples wide, of the percentages covered
        # by the bounded or unbounded distributions.
        self.percentages = inputs[:, 0].reshape(self.length, 1) + np.arange(0,
                                                                            self.samples, 1) * inputs[:, 1].reshape(self.length, 1)
        # Transform percentages into a sample space
        self.hypercube = np.zeros((self.length, self.samples))
        for uncertainty_type in self.choices:
            mask = self.params['uncertainty_type'] == uncertainty_type.id
            if not mask.sum():
                continue
            self.hypercube[mask, :] = \
                uncertainty_type.ppf(params=self.params[mask],
                                     percentages=self.percentages[mask, :])
        self.hypercube[self.params['negative'], :] = \
            -1 * self.hypercube[self.params['negative'], :]

    def next(self):
        """Draw directly from pre-computed sample space."""
        return self.hypercube[self.row_index,
                              self.random.randint(self.samples, size=self.length)]
