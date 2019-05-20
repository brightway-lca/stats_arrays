Documentation for ``stats_arrays``
==================================

The ``stats_arrays`` package provides a standard NumPy array interface for defining uncertain parameters used in models, and classes for Monte Carlo sampling. It also plays well with others.

Motivation
==========

* Want a consistent interface to SciPy and NumPy statistical function
* Want to be able to quickly load and save many parameter uncertainty distribution definitions in a portable format
* Want to manipulate and switch parameter uncertainty distributions and variables
* Want simple Monte Carlo random number generators that return a vector of parameter values to be fed into uncertainty or sensitivity analysis
* Want something simple, extensible, documented and tested

The ``stats_arrays`` package was originally developed for the `Brightway2 life cycle assessment framework <http://brightwaylca.org/>`_, but can be applied to any stochastic model.

Example
=======

.. code-block:: python

    >>> from stats_arrays import *
    >>> my_variables = UncertaintyBase.from_dicts(
    ...     {'loc': 2, 'scale': 0.5, 'uncertainty_type': NormalUncertainty.id},
    ...     {'loc': 1.5, 'minimum': 0, 'maximum': 10, 'uncertainty_type': TriangularUncertainty.id}
    ... )
    >>> my_variables
    array([(2.0, 0.5, nan, nan, nan, False, 3),
           (1.5, nan, nan, 0.0, 10.0, False, 5)],
        dtype=[('loc', '<f8'), ('scale', '<f8'), ('shape', '<f8'),
               ('minimum', '<f8'), ('maximum', '<f8'), ('negative', '?'),
               ('uncertainty_type', 'u1')])
    >>> my_rng = MCRandomNumberGenerator(my_variables)
    >>> my_rng.next()
    array([ 2.74414022,  3.54748507])
    >>> # can also be used as an interator
    >>> zip(my_rng, xrange(10))
    [(array([ 2.96893108,  2.90654471]), 0),
     (array([ 2.31190619,  1.49471845]), 1),
     (array([ 3.02026168,  3.33696367]), 2),
     (array([ 2.04775418,  3.68356226]), 3),
     (array([ 2.61976694,  7.0149952 ]), 4),
     (array([ 1.79914025,  6.55264372]), 5),
     (array([ 2.2389968 ,  1.11165296]), 6),
     (array([ 1.69236527,  3.24463981]), 7),
     (array([ 1.77750176,  1.90119991]), 8),
     (array([ 2.32664152,  0.84490754]), 9)]

See a `more complete notebook example <http://nbviewer.ipython.org/url/brightwaylca.org/examples/stats-arrays-demo.ipynb>`_.

.. _params-array:

Parameter array
===============

The core data structure for ``stats_arrays`` is a parameter array, which is made from a special kind of NumPy array called a `NumPy structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ which has the following data type:

.. code-block:: python

    import numpy as np
    base_dtype = [
        ('loc', np.float64),
        ('scale', np.float64),
        ('shape', np.float64),
        ('minimum', np.float64),
        ('maximum', np.float64),
        ('negative', np.bool)
    ]

.. note::

    Read more on `NumPy data types <http://docs.scipy.org/doc/numpy/user/basics.types.html>`_.

.. note::

    The *negative* column is used for uncertain parameters whose distributions are normally always positive, such as the lognormal, but in this case have negative values.

In general, most uncertainty distributions can be defined by three variables, commonly called *location*, *scale*, and *shape*. The *minimum* and *maximum* values make distributions **bounded**, so that one can, for example, define a normal uncertainty which is always positive.

.. warning:: Bounds are not applied in the following methods: 1) Distribution functions (``PDF``, ``CDF``, etc.) where you supply the input vector. 2) ``.statistics``, which gives 95 percent confidence intervals for the unbounded distribution.

.. _hpa:

Heterogeneous parameter array
-----------------------------

Parameter arrays can have multiple uncertainty distributions. To distinguish between the different distributions, another column, called ``uncertainty_type``, is added:

.. code-block:: python

    heterogeneous_dtype = [
        ('uncertainty_type', np.uint8),
        ('loc', np.float64),
        ('scale', np.float64),
        ('shape', np.float64),
        ('minimum', np.float64),
        ('maximum', np.float64),
        ('negative', np.bool)
    ]

Note that `stats_arrays` was developed in conjunction with the `Brightway LCA framework <https://brightwaylca.org/>`__; Brightway uses the field name "uncertainty type", without the underscore. Be sure to use the underscore when using `stats_arrays`.

Each uncertainty distribution has an integer ID number. See the table below for built-in distribution IDs.

.. note::

    The recommended way to use uncertainty distribution IDs is not by looking up the integers manually, but by referring to ``SomeClass.id``, e.g. ``LognormalDistribution.id``.

Mapping parameter array columns to uncertainty distributions
------------------------------------------------------------

======================= === =========================== ============================= =========================== =============== ===============
Name                    ID  ``loc``                     ``scale``                     ``shape``                   ``minimum``             ``maximum``
======================= === =========================== ============================= =========================== =============== ===============
Undefined               0   **static value**
No uncertainty          1   **static value**
:ref:`lognormal` [#]_   2   :math:`\boldsymbol{\mu}`    :math:`\boldsymbol{\sigma}`                               *lower bound*   *upper bound*
:ref:`normal` [#]_      3   :math:`\boldsymbol{\mu}`    :math:`\boldsymbol{\sigma}`                               *lower bound*   *upper bound*
:ref:`uniform` [#]_     4                                                                                         *minimum* [#]_  **maximum**
:ref:`triangular` [#]_  5   *mode* [#]_                                                                           *minimum* [#]_  **maximum**
:ref:`bernoulli` [#]_   6   **p**                                                                                 *lower bound*   *upper bound*
:ref:`discreteu` [#]_   7                                                                                         *minimum* [#]_  **upper bound** [#]_
:ref:`weibull` [#]_     8   *offset* [#]_               :math:`\boldsymbol{\lambda}`  :math:`\boldsymbol{k}`
:ref:`gamma` [#]_       9   *offset* [#]_               :math:`\boldsymbol{\theta}`   :math:`\boldsymbol{k}`
:ref:`beta` [#]_        10  :math:`\boldsymbol{\alpha}`                               :math:`\boldsymbol{\beta}`  *lower bound*           *upper bound*
:ref:`extreme` [#]_     11  :math:`\boldsymbol{\mu}`    :math:`\boldsymbol{\sigma}`   :math:`\boldsymbol{\xi}`
:ref:`students` [#]_    12  *median*                    *scale*                       :math:`\boldsymbol{\nu}`
======================= === =========================== ============================= =========================== =============== ===============

Items in **bold** are required, items in *italics* are optional.

.. [#] `Lognormal distribution <http://en.wikipedia.org/wiki/Log-normal_distribution>`_. :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the underlying normal distribution
.. [#] `Normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
.. [#] `Uniform distribution <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_
.. [#] Default is 0 if not otherwise specified
.. [#] `Triangular distribution <https://en.wikipedia.org/wiki/Triangular_distribution>`_
.. [#] Default is :math:`(minimum + maximum) / 2`
.. [#] Default is 0 if not otherwise specified
.. [#] `Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_. If ``minimum`` **and** ``maximum`` are specified, :math:`p` is not limited to :math:`0 < p < 1`, but instead to the interval :math:`(minimum,maximum)`
.. [#] `Discrete uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(discrete)>`_
.. [#] The discrete uniform operates on a "half-open" interval, :math:`[minimum, maximum)`, where the minimum is included but the maximum is not. Default is 0 if not otherwise specified.
.. [#] The distribution includes values up to, but not including, the ``maximum``.
.. [#] `Weibull distribution <https://en.wikipedia.org/wiki/Weibull_distribution>`_
.. [#] Optional offset from the origin
.. [#] `Gamma distribution <https://en.wikipedia.org/wiki/Gamma_distribution>`_
.. [#] Optional offset from the origin
.. [#] `Beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`_
.. [#] `Extreme value distribution <https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution>`_
.. [#] `Student's T distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_

Unused columns can be given any value, but it is recommended that they are set to ``np.NaN``.

.. warning::

    Unused optional columns **must** be set to ``np.NaN`` to avoid unexpected behaviour!

Extending parameter arrays
--------------------------

Parameter arrays can have additional columns. For example, model parameters that will be inserted into a matrix could have columns called *row* and *column*. For speed reasons, it is recommended that only NumPy numeric types are used if the arrays are to stored on disk.

Technical reference
===================

Probability distributions
-------------------------

.. toctree::
   :maxdepth: 1

   distributions/base
   distributions/lognormal
   distributions/normal
   distributions/uniform
   distributions/discrete-uniform
   distributions/triangular
   distributions/bernoulli
   distributions/beta
   distributions/extreme
   distributions/student
   distributions/gamma
   distributions/weibull

Random number generators
------------------------

.. toctree::
   :maxdepth: 1

   rng
   mcrng
   lhc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

