# stats_arrays Changelog

# 3.0 (2026-09-02)

### Breaking changes

* `statistics()` now returns Python floats throughout. Several distributions previously returned single-element numpy arrays, so callers relying on `result["mean"]` being an array, or on `result["mean"] == x` evaluating to `array([True])` rather than a bool, will see the difference. The base class has always documented the intended behaviour: "All values should be floats (not single-element arrays)."

* `statistics()` no longer returns placeholder strings. **Beta** returned `"Not Implemented"` for `median`, `lower` and `upper`, and `"Undefined"` for `mode` when alpha or beta was at most 1; **DiscreteUniform** returned `"Undefined"` for `mode`. Undefined values are now `None`, as the base class has always documented.

### Bug fixes

* **DiscreteUniform** — `statistics()` reported a `mean` of `(minimum + maximum) / 2` and a `median` of that value rounded to an integer. The support excludes `maximum`, so both were off by one half: for the values 5 to 9 the mean was 7.5 and the median 8, where both are 7. Both now equal `(minimum + maximum - 1) / 2`, matching `scipy.stats.randint`, and the median is a float that lands on a half-integer for an even count of values.
* **Beta** and **BetaPERT** — `statistics()` now reports `median`, `lower` and `upper` (the 2.5th and 97.5th percentiles), all from the `ppf`. The mode is now placed on the lower bound when alpha ≤ 1 < beta and on the upper bound when beta ≤ 1 < alpha, rather than being called undefined; it is `None` only for the flat (alpha = beta = 1) and bimodal (alpha, beta < 1) cases.
* **Lognormal** and **DiscreteUniform** — `statistics()` raised `TypeError: only 0-dimensional arrays can be converted to Python scalars` on numpy 2, since both called `float()` on a one-element array. Normal and Triangular already used `.flat[0]`; the others now do too.
* **UncertaintyBase** — the inherited `statistics()` returned `params["loc"]` as an array, so every distribution that did not override it (Undefined, NoUncertainty, Bernoulli, Weibull, Gamma, GeneralizedExtremeValue, StudentsT) reported an array `mean`.
* **Uniform** — `statistics()` returned all five values as arrays.

# 2.0 (2026-05-15)

### Breaking changes

* `seeded_random` arguments to `random_variables` and `bounded_random_variables` no longer accept plain integers. Pass `np.random.RandomState(seed)` explicitly, or omit the argument entirely. Passing an `int` now raises `TypeError` with a clear message. Note: the high-level RNG classes (`MCRandomNumberGenerator`, `LatinHypercubeRNG`, `RandomNumberGenerator`) are unaffected — they accept an integer `seed` as always.

### New features

* Added `UncertaintyType` IntEnum mapping distribution names to their integer IDs (`UncertaintyType.normal`, `UncertaintyType.lognormal`, etc.). Fully backwards-compatible — members compare equal to plain ints.
* `validate()` error messages now include the specific row indices that failed (e.g. `Failing rows: [1, 3]`), making it much easier to diagnose problems in large parameter arrays.
* Added `notebooks/stats_arrays_demo.ipynb`, a runnable example notebook covering params array construction, all RNG classes, PDF/CDF/PPF inspection, and a worked Monte Carlo propagation example.

### Bug fixes

* **Lognormal** — `pdf()` default x-axis was computed using μ instead of exp(μ), producing a wildly incorrect range.
* **Bernoulli** — `cdf()` and `ppf()` were logically inverted: `P(X=0)` and `P(X=1)` were swapped.
* **DiscreteUniform** — `cdf()` called `scipy.stats.randint` with `loc`/`scale` keyword arguments instead of the required positional `low`/`high` shape parameters, returning wrong probabilities.
* **Weibull** and **GeneralizedExtremeValue** — `validate()` did not check for `NaN` scale/shape values; `NaN <= 0` is `False` so invalid params silently passed.
* **TriangularUncertainty.pdf()** — three bugs in the default (no `xs`) branch: `if not mode` treated `mode=0.0` as falsy and replaced it with the midpoint; the peak height formula returned the normalised mode position instead of `2/(upper-lower)`; and `mode==minimum` or `mode==maximum` produced duplicate x-coordinates and an all-zero y array.

# 1.0.1 (2025-11-12)

* Added `BetaPERTUncertainty` to `uncertainty_choices`

# 1.0 (2025-10-27)

* Added types
* **BREAKING CHANGE**: Bernoulli distribution is now correctly strictly limited to outputting 0 or 1; `minimum` and `maximum` are ignored.
* Fixed triangular distribution `.pdf` method which didn't account for the minimum to maximum range but gave values for a `(0, 1)` range.
* Added many more tests

## 0.8 (2025-10-23)

* [#19: Add Beta PERT distribution](https://github.com/brightway-lca/stats_arrays/pull/19).
* Moved to a `src` layout.
* Switch to absolute imports

## 0.7 (2024-08-19)

* Moved to `pyproject.toml` packaging and `pytest` tests
* Numpy 2.0 compatiblity

### 0.6.6 (2023-10-18)

* Merged [PR #13](https://github.com/brightway-lca/stats_arrays/pull/13). Move argsort to save on execution time. Thanks @Loisel!
* Merged [PR #12](https://github.com/brightway-lca/stats_arrays/pull/12). Update links in readme. Thanks @mfastudillo!
* Merged [PR #10](https://github.com/brightway-lca/stats_arrays/pull/10). Fix flaky test on distributions::extreme.py, distributions::gama.py, and distributions::student.py. Thanks @lonly7star!

### 0.6.5 (2021-05-06)

* Updates for changes in Numpy API

### 0.6.4 (2020-01-31)

* Merged [PR #5](https://bitbucket.org/cmutel/stats_arrays/pull-requests/5/use-meanround-0-instead-of-round-mean/diff), fix rounding function.

### 0.6.3 (2019-11-29)

* Improve speed of lognormal CDF and PPF under common conditions

### 0.6.2 (2019-11-13)

* Restore Python 2.7 compatibility

### 0.6.1 (2019-07-29)

* Several bug fixes from Daniel de Koning
* Start improvement of beta distribution

## 0.6 (2019-05-19)

* Allow `MCRandomNumberGenerator` to generate multiple samples at once
* Move tests to pytest
* Add tests for MCRNG

### 0.5.1 (2019-05-11)

* Import from `collections.abc` when possible, for Python 3.8 support.
