from __future__ import division
from ..errors import InvalidParamsError,\
    ImproperBoundsError, UnreasonableBoundsError, \
    MaximumIterationsError
from ..utils import one_row_params_array, construct_params_array
import numpy as np

np.seterr(invalid='ignore')


class UncertaintyBase(object):
    """
Abstract base class for uncertainty types.

All methods on uncertainty classes should be `class methods <http://docs.python.org/library/functions.html#classmethod>`_, as instantiating uncertainty classes many times is not desired.

.. rubric:: Defaults

* ``default_number_points_in_pdf``: 200. The default number of points to calculate for PDF/CDF functions.
* ``standard_deviations_in_default_range``: 3. The number of standard deviations that define the default range when calculating PDF/CDF values. In a normal distribution, 3 standard deviations is approximately 99% of all values.

"""
    default_number_points_in_pdf = 200
    standard_deviations_in_default_range = 2.2

    # Conversion utilities ###
    @classmethod
    def from_tuples(cls, *data):
        """
Construct a :ref:`hpa` from parameter tuples.

The order of the parameters is:

#. ``loc``
#. ``scale``
#. ``shape``
#. ``minimum``
#. ``maximum``
#. ``negative``
#. ``uncertainty_type``

Each input tuple must have a length of exactly 7. For more flexibility, use ``from_dicts``.

Example:

.. code-block:: python

    >>> from stats_arrays import UncertaintyBase
    >>> import numpy as np
    >>> UncertaintyBase.from_tuples(
    ...     (2, 3, np.NaN, np.NaN, np.NaN, False, 3),
    ...     (5, np.NaN, np.NaN, 3, 10, False, 5)
    ...     )
    array([(2.0, 3.0, nan, nan, nan, False, 3),
           (5.0, nan, nan, 3.0, 10.0, False, 5)],
           dtype=[('loc', '<f8'), ('scale', '<f8'), ('shape', '<f8'),
                  ('minimum', '<f8'), ('maximum', '<f8'), ('negative', '?'),
                  ('uncertainty_type', 'u1')])

Args:
    One of more tuples of length 7.

Returns:
    A :ref:`hpa`

        """
        params = construct_params_array(len(data), True)
        for index, row in enumerate(data):
            params[index] = row
        return params

    @classmethod
    def from_dicts(cls, *dicts):
        """
Construct a :ref:`hpa` from parameter dictionaries.

Dictionary keys are the normal parameter array columns. Each distribution defines which columns are required and which are optional.

Example:

.. code-block:: python

    >>> from stats_arrays import UncertaintyBase
    >>> import numpy as np
    >>> UncertaintyBase.from_dicts(
    ...     {'loc': 2, 'scale': 3, 'uncertainty_type': 3},
    ...     {'loc': 5, 'minimum': 3, 'maximum': 10, 'uncertainty_type': 5}
    ...     )
    array([(2.0, 3.0, nan, nan, nan, False, 3),
           (5.0, nan, nan, 3.0, 10.0, False, 5)],
           dtype=[('loc', '<f8'), ('scale', '<f8'), ('shape', '<f8'),
                  ('minimum', '<f8'), ('maximum', '<f8'), ('negative', '?'),
                  ('uncertainty_type', 'u1')])

Args:
    One of more dictionaries.

Returns:
    A :ref:`hpa`

        """
        LABELS = [
            ('loc', np.NaN),
            ('scale', np.NaN),
            ('shape', np.NaN),
            ('minimum', np.NaN),
            ('maximum', np.NaN),
            ('negative', False),
            ('uncertainty_type', 0),
        ]
        params = construct_params_array(len(dicts), True)
        for index, obj in enumerate(dicts):
            data = [obj.get(key, default) for key, default in LABELS]
            if 'uncertainty type' in obj and 'uncertainty_type' not in obj:
                data[-1] = obj['uncertainty type']
            params[index] = tuple(data)
        return params

    # Utility methods ###
    @classmethod
    def validate(cls, params):
        """
Validate the parameter array for uncertainty distribution.

Validation is distribution specific. The only default check is that ``minimum`` is less than or equal to ``maximum``, and otherwise raises ``stats_arrays.ImproperBoundsError``.

Doesn't return anything.

Args:
    A :ref:`params-array`.

    """
        # Minimum <= Maximum
        if (params['minimum'] >= params['maximum']).sum():
            raise ImproperBoundsError

    @classmethod
    def check_2d_inputs(cls, params, vector):
        """Convert ``vector`` to 2 dimensions if not already, and raise ``stats_arrays.InvalidParamsError`` if ``vector`` and ``params`` dimensions don't match."""
        if len(vector.shape) == 1:
            # Slices from structured arrays can't always be resized
            vector = np.array(vector)
            # Transform to 2-dimensional
            vector.resize(vector.shape[0], 1)
        if params.shape[0] != vector.shape[0]:
            raise InvalidParamsError(
                "Vector shape must be either (m,) or (m,n), where params has m rows. Vector has shape %s, and params is %i rows" %
                (vector.shape, params.shape[0]))
        return vector

    @classmethod
    @one_row_params_array
    def check_bounds_reasonableness(cls, params, threshold=0.1):
        """
Test if there is at least a ``threshold`` percent chance of generating random numbers within the provided bounds.

Only works is **both min and max** bounds are defined!

Doesn't return anything. Raises ``stats_arrays.UnreasonableBoundsError`` if this condition is not met.

.. rubric:: Inputs

* params : A one-row :ref:`params-array`.
* threshold : A percentage between 0 and 1. The minimum loc of the distribution covered by the bounds before an error is raised."""
        min_percentage = float(cls.cdf(params, params['minimum']))
        max_percentage = float(cls.cdf(params, params['maximum']))
        coverage = max_percentage - min_percentage
        if coverage < threshold:
            raise UnreasonableBoundsError(
                "The provided bounds cover only %.2f percent of the possible distribution" % coverage)

    # Used for Monte Carlo ###
    @classmethod
    def bounded_random_variables(cls, params, size, seeded_random=None,
                                 maximum_iterations=50):
        """Generate random variables repeatedly until all varaibles are within the bounds of each distribution. Raise MaximumIterationsError if this takes more that `maximum_iterations`. Uses `random_variables` for random number generation.

.. rubric:: Inputs

* params : A :ref:`params-array`.
* size : Integer. The number of values to draw from each distribution in `params`.
* seeded_random : Integer. Optional. Random seed to get repeatable samples.
* maximum_iterations : Integer. Optional. Maximum iterations to try to fit the given bounds before an error is raised.

.. rubric:: Output

An array of random values, with dimensions `params` rows by `size`."""
        data = cls.random_variables(params, size, seeded_random)
        min_array = params['minimum'].reshape(params.shape[0], 1)
        max_array = params['maximum'].reshape(params.shape[0], 1)
        # Check bounds => boolean array where True is out of bounds
        # All NaN values will evaluate to false in a comparison;
        # No special handling needed for unbounded values
        # bounds_mask is 2 dimensional
        bounds_mask = (data < min_array) + (data > max_array)
        counter = 0
        while bounds_mask.sum() > 0:
            # This isn't the most efficient, but random number generation
            # is fast, so pass the whole params. The problem is that data is
            # a large multi-dimensional array, and there is no easy way to pick
            # the values that should be regenerated. Testing shows that this
            # approach increases execution speed linearly as a function of the
            # most restrictive bounds, as all the other bounds are satisfied
            # during iteration for the worst-case scenario. As this approach is
            # O(n), and the time for random number generation is << time for
            # solving the linear system, we don't try to find a clever way
            # around this inefficiency. See stats/tests/uncertainty.py -
            # UncertaintyTestCase - test_random_timing for a timing test.
            data[bounds_mask] = cls.random_variables(params,
                                                     size, seeded_random)[bounds_mask]
            bounds_mask = (data < min_array) + (data > max_array)

            counter += 1
            if counter >= maximum_iterations:
                raise MaximumIterationsError
        return data

    @classmethod
    def random_variables(cls, params, size, seeded_random=None):
        """Generate random variables for the given uncertainty. Should **not check** to ensure that random samples are with the (minimum, maximum bounds). Bounds checking is provided by the `bounded_random_variables` class method.

.. rubric:: Inputs

* params : A :ref:`params-array`.
* size : Integer. The number of values to draw from each distribution in `params`.
* seeded_random : Integer. Optional.

.. rubric:: Output

An array of random values, with dimensions `params` rows by `size`."""
        raise NotImplementedError

    # Used for Latin Hypercube Monte Carlo ###
    @classmethod
    def ppf(cls, params, percentages):
        """Return percent point function (inverse of CDF, e.g. value in distribution where x percent of the distribution is less than value) for various distributions.

.. rubric:: Inputs

* params : A :ref:`params-array`.
* percentages : An array of percentages, bounded on (0,1). Each row in `percentages` corresponds to a row in `params`.

.. rubric:: Output

An array of values within the ranges of each distribtion, with `params` rows and `percentages` columns."""
        percentages = cls.check_ppf_inputs(params, percentages)
        raise NotImplementedError

    @classmethod
    def cdf(cls, params, vector):
        """Used when a distribution is bounded, to determine where to begin or end the percentages used in calculating hypercube sampling space.

.. rubric:: Inputs

* params : A :ref:`params-array`.
* vector : A array of values taken from the uncertainty distributions, with **one row** or the **same number** of rows as `params`.

.. rubric:: Output

An array of cumulative densities, bounded on (0,1), with `params` rows and `vector` columns."""
        vector = cls.check_cdf_inputs(params, vector)
        raise NotImplementedError

    # Used for graphing ###
    @classmethod
    @one_row_params_array
    def statistics(cls, params):
        """Build a dictionary of mean, mode, median, and 95% confidence interval upper and lower values.

.. rubric:: Inputs

* params : A one-row :ref:`params-array`.

.. rubric:: Output

{'mean': mean value, 'mode': mode value, 'median': median value, 'upper': upper limit value, 'lower': lower limit value}. All values should be floats (not single-element arrays). Parameters that are not defined should be returned `None`, not omitted.
        """
        return {'mean': params['loc'], 'mode': None, 'median': None,
                'upper': None, 'lower': None}

    @classmethod
    @one_row_params_array
    def pdf(cls, params, xs=None):
        """Provide a standard interface to calculate the probability distribution function of a uncertainty distribution. Default is `cls.default_number_points_in_pdf` points between min to max range if bounds are present, or `cls.standard_deviations_in_default_range` standard distributions.

.. rubric:: Inputs

* params : A one-row :ref:`params-array`.
* xs : Optional. A one-dimensional numpy array of input values.

.. rubric:: Output

.. important:: The output format for PDF is different than CDF or PPF.

A tuple of a vactor x values and a vector of y values. Y values are a one-dimensional array of probability densities, bounded on (0,1), with length `xs`, if provided, or `cls.default_number_points_in_pdf`."""
        raise NotImplementedError


class BoundedUncertaintyBase(UncertaintyBase):

    """An uncertainty distribution where minimum and maximum bounds are required. No bounds checking is required for these distributions, as bounds are integral inputs into the sample space generator."""
    @classmethod
    def validate(cls, params):
        super(BoundedUncertaintyBase, cls).validate(params)
        if np.isnan(params['minimum']).sum() or np.isnan(params['maximum']).sum():
            raise ImproperBoundsError("This distribution require minimum and maximum values.")

    @classmethod
    def rescale(cls, params):
        """Rescale params to a (0,1) interval. Return adjusted_means and scale. Needed because SciPy assumes a (0,1) interval for many distributions."""
        scale = (params['maximum'] - params['minimum'])
        adjusted_means = (params['loc'] - params['minimum']) / scale
        return adjusted_means, scale

    @classmethod
    def bounded_random_variables(cls, params, size, seeded_random=None,
                                 maximum_iterations=None):
        """No bounds checking because the bounds do not exclude any of the distribution."""
        return cls.random_variables(params, size, seeded_random)

    @classmethod
    @one_row_params_array
    def check_bounds_reasonableness(cls, params):
        """Always true because the bounds do not exclude any of the distribution."""
        return
