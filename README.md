The `stats_arrays` package provides a standard NumPy array interface for defining uncertain parameters used in models, and classes for Monte Carlo sampling. It also plays well with others.

# Motivation

* Want a consistent interface to SciPy and NumPy statistical function
* Want to be able to quickly load and save many parameter uncertainty distribution definitions in a portable format
* Want to manipulate and switch parameter uncertainty distributions and variables
* Want simple Monte Carlo random number generators that return a vector of parameter values to be fed into uncertainty or sensitivity analysis
* Want something simple, extensible, documented and tested

The `stats_arrays package was originally developed for the [Brightway2 life cycle assessment framework](http://brightwaylca.org/), but can be applied to any stochastic model.

# Example

```python

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

```

# More

* Source code: https://bitbucket.org/cmutel/stats_arrays
* Online documentation: https://stats_arrays.readthedocs.io/en/latest/
