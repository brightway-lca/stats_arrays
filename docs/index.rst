Documentation for ``stats_arrays``
==================================

The ``stats_arrays`` package provides a common interface for defining uncertain parameters used in models, and classes for sampling the parameters using Monte Carlo sampling or similar random number generation.

Motivation
==========

* Some uncertainty distributions are defined differently in SciPy and NumPy (e.g. lognormal), and a common interface would make life easier and reduce errors.
* Want to be able to quickly load and save model parameter uncertainty distribution definitions in a portable format.
* Want to store model parameter uncertainty distribution definitions in NumPy arrays to allow for easy manipulation, e.g. quickly replace one distribution with another.
* Want simple Monte Carlo (and Monte Carlo subclasses) random number generators that return a vector of parameter values to be fed into uncertainty or sensitivity analysis.

The ``stats_arrays`` package was originally developed for the `Brightway2 life cycle assessment framework <http://brightwaylca.org/>`_, but can be applied to any stochastic model.

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

Each uncertainty distribution has an integer ID number. See the table below for built-in distribution IDs.

.. note::

    The recommended way to use uncertainty distribution IDs is not by looking up the integers manually, but by referring to ``SomeClass.id``, e.g. ``LognormalDistribution.id``.

Mapping parameter array columns to uncertainty distributions
------------------------------------------------------------

======================= === =========================== =========================== =============== =============== ===============
Name                    ID  ``loc``                     ``scale``                   ``shape``       ``minimum``             ``maximum``
======================= === =========================== =========================== =============== =============== ===============
Undefined               0   **static value**
No uncertainty          1   **static value**
:ref:`lognormal` [#]_   2   :math:`\boldsymbol{\mu}`    :math:`\boldsymbol{\sigma}`                 *lower bound*   *upper bound*
:ref:`normal` [#]_      3   :math:`\boldsymbol{\mu}`    :math:`\boldsymbol{\sigma}`                 *lower bound*   *upper bound*
Uniform [#]_            4                                                                           *minimum* [#]_  **maximum**
Triangular [#]_         5   *mode* [#]_                                                             *minimum* [#]_  **maximum**
:ref:`bernoulli` [#]_   6   **p**                                                                   *lower bound*   *upper bound*
Discrete uniform [#]_   7                                                                           *minimum* [#]_  **upper bound**
:ref:`beta`             10  :math:`\boldsymbol{\alpha}` :math:`\boldsymbol{\beta}`
======================= === =========================== =========================== =============== =============== ===============

Items in **bold** are required, items in *italics* are optional.

.. [#] `Lognormal distribution <http://en.wikipedia.org/wiki/Log-normal_distribution>`_. :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the underlying normal distribution.
.. [#] `Normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
.. [#] `Uniform distribution <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_
.. [#] Default is 0 if not otherwise specified
.. [#] `Triangular distribution <https://en.wikipedia.org/wiki/Triangular_distribution>`_
.. [#] Default is :math:`(minimum + maximum) / 2`
.. [#] Default is 0 if not otherwise specified
.. [#] `Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_. If ``minimum`` **and** ``maximum`` are specified, :math:`p` is not limited to :math:`0 < p < 1`, but instead to the interval :math:`(minimum,maximum)`.
.. [#] `Discrete uniform <https://en.wikipedia.org/wiki/Uniform_distribution_(discrete)>`_.
.. [#] Default is 0 if not otherwise specified

Unused columns can be given any value, but it is recommended that they are set to ``np.NaN``.

.. warning::

    The *minimum* and *maximum* columns must be set to ``np.NaN`` if no bounds are desired.

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
   distributions/triangular
   distributions/bernoulli
   distributions/beta

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

