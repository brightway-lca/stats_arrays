Monte Carlo random number generator
***********************************

.. autoclass:: stats_arrays.MCRandomNumberGenerator
    :members:

    .. automethod:: stats_arrays.MCRandomNumberGenerator.__init__

Choosing a random number source
===============================

The random number generators draw from a ``numpy.random.Generator``. ``seed`` accepts:

* ``None`` (default): an unseeded ``Generator``.
* An integer, sequence of integers, ``SeedSequence`` or ``BitGenerator``: passed to ``numpy.random.default_rng``.
* A ``numpy.random.Generator`` or ``numpy.random.RandomState`` instance: used as is.

.. code-block:: python

    >>> import numpy as np
    >>> mcrng = MCRandomNumberGenerator(params, seed=42)  # default_rng(42)
    >>> mcrng = MCRandomNumberGenerator(params, seed=np.random.default_rng(42))

For parallel workers, give each its own stream with ``SeedSequence.spawn``:

.. code-block:: python

    >>> children = np.random.SeedSequence(42).spawn(n_workers)
    >>> mcrng = MCRandomNumberGenerator(params, seed=children[worker_index])

NumPy keeps the raw bit streams of a ``Generator`` stable, but may change how individual distributions are sampled between NumPy versions (`NEP 19 <https://numpy.org/neps/nep-0019-rng-policy.html>`_). To reproduce results exactly in the long term, record the NumPy version along with the seed.

Reproducing results from stats_arrays 4 and earlier
---------------------------------------------------

Before version 5, an integer ``seed`` seeded the legacy ``numpy.random.RandomState``, so the same integer now gives different numbers. To get the old numbers back, pass a ``RandomState`` instead of the integer:

.. code-block:: python

    >>> # stats_arrays 4: MCRandomNumberGenerator(params, seed=42)
    >>> mcrng = MCRandomNumberGenerator(params, seed=np.random.RandomState(42))

This gives exactly the same values as before, for ``MCRandomNumberGenerator``, ``RandomNumberGenerator`` and ``LatinHypercubeRNG``. It is also about three times slower than a ``Generator`` for large lognormal or normal parameter arrays.

Avoiding per-draw allocations
=============================

For large parameter arrays drawn many times, ``generate`` can write into a preallocated array, and can skip putting the values back into the original row order:

.. code-block:: python

    >>> buffer = np.empty(len(params))
    >>> mcrng.generate(out=buffer)                   # original row order
    >>> mcrng.generate(out=buffer, order="sorted")   # internal order

In ``"sorted"`` order, value ``i`` belongs to row ``mcrng.ordering[i]`` of the original ``params``. Code that already keeps an index array for the parameters (for example, matrix positions) can apply ``ordering`` to that array once, and then use the sorted values directly on every draw.
