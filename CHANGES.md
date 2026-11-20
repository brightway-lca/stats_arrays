# stats_arrays Changelog

# Unreleased

### Breaking changes

* **Python 3.11 is now the minimum supported version.** Python 3.9 and 3.10 have reached end of life, and current NumPy and SciPy releases no longer support them. See [brightway2#92](https://github.com/brightway-lca/brightway2/issues/92). Python 3.15 is now tested in CI.
* **Random numbers now come from `numpy.random.Generator`** ([#46](https://github.com/brightway-lca/stats_arrays/issues/46)). In `MCRandomNumberGenerator`, `RandomNumberGenerator` and `LatinHypercubeRNG`, an integer `seed` is now passed to `numpy.random.default_rng` instead of `numpy.random.RandomState`, and no seed gives an unseeded `Generator`. The same integer seed therefore gives different numbers than in 4.x. The unseeded fallback in each distribution's `random_variables` is now `default_rng()` too.
  * **To get the old numbers back**, pass a `RandomState` instead of the integer: `MCRandomNumberGenerator(params, seed=numpy.random.RandomState(42))` gives exactly what `MCRandomNumberGenerator(params, seed=42)` gave in 4.x. The same works for `RandomNumberGenerator` and `LatinHypercubeRNG`.
  * NumPy keeps `Generator` bit streams stable but may change how individual distributions are sampled between NumPy versions (NEP 19). Record the NumPy version with the seed if you need exact long-term reproducibility.

### New features

* `seed` accepts an integer, a sequence of integers, a `SeedSequence`, a `BitGenerator`, or a `Generator` or `RandomState` instance (used as is). Use `SeedSequence(seed).spawn(n)` for independent streams in parallel workers. `seeded_random` in `bounded_random_variables`/`random_variables` accepts a `Generator` as well as a `RandomState`.
* **MCRandomNumberGenerator.generate()** takes `out=`, a preallocated array to write into, and `order="sorted"`, which returns values in the internal order (sorted by `uncertainty_type`, row `i` belongs to `ordering[i]` of the input) and skips the permutation back to the input order.
* New public helpers in `stats_arrays.utils`: `make_random_source(seed)`, `random_integers(random, low, high, size)`, and the `RandomSource` and `SeedLike` type aliases. `UncertaintyBase.has_bounds(params)` reports whether any row sets `minimum` or `maximum`.

### Bug fixes

* **BernoulliUncertainty.random_variables()** raised `ValueError` for more than one row, because the uniform draws of shape `(rows, size)` were compared against `loc` of shape `(rows,)`.
* **StudentsTUncertainty.random_variables()** broadcast `loc` and `scale` against the draws incorrectly. With several rows it raised `ValueError` for `size > 1`, and for `size == 1` every row used the `loc` and `scale` of the first row. Seeded results for parameter arrays with several Student's T rows that differ in `loc` or `scale` therefore change.
* **LatinHypercubeRNG** ignored `minimum` and `maximum` bounds ([#48](https://github.com/brightway-lca/stats_arrays/issues/48)): the adjusted percentages were assigned to a temporary copy, so bounded distributions were sampled over their full range. The hypercube now only covers the part of each distribution inside its bounds. This changes the samples for every row that has a bound and is not one of the distributions where `minimum` and `maximum` are the support (uniform, triangular, discrete uniform, beta and beta-PERT, which are unchanged). Details:
  * A lower bound starts at P(X < minimum), so discrete distributions can still return `minimum` itself.
  * For a lognormal with `negative` set, the bounds apply to the signed values.
  * Undefined and no-uncertainty rows with bounds no longer raise `UndefinedDistributionError`; their bounds are ignored, as their value is always `loc`.
  * Bounds that contain none of the distribution, or so little that it is lost to rounding, raise `UnreasonableBoundsError` with the failing rows.
  * Distributions without `ppf` (Weibull, gamma, generalized extreme value and Student's t), or without `cdf` on rows with bounds, raise `NotImplementedError` naming the distribution and the failing rows. Before, the error had no message.
* **Normal** and **Lognormal** `validate()` accepted `minimum >= maximum` ([#62](https://github.com/brightway-lca/stats_arrays/issues/62)), as they did not run the base-class bounds check. They now raise `ImproperBoundsError` with the failing rows, so `MCRandomNumberGenerator` and `LatinHypercubeRNG` reject such rows at instantiation. Before, Monte Carlo failed at the first draw with a `MaximumIterationsError`.
* **LatinHypercubeRNG** negated rows with `negative` set after `ppf` ([#54](https://github.com/brightway-lca/stats_arrays/issues/54)). `LognormalUncertainty.ppf` already negates them, so negative lognormals came out positive, and other distributions were flipped although `MCRandomNumberGenerator` leaves them alone. The sign is now left to each distribution's `ppf`. Latin hypercube samples change for every row with `negative` set: negative lognormals are now negative, and for the other distributions that have a `ppf` the flag no longer has an effect, as in Monte Carlo. Gamma and Weibull honour `negative` in Monte Carlo, but have no `ppf`, so the Latin hypercube cannot sample them.
* The six bounded-interval tests in `tests/test_lhc.py` had been skipped since 2019, which hid this bug. They run again; the three lognormal ones had expected values for `loc = 0` but used `loc = 1`, and now use `loc = 0`.
* **UncertaintyChoices.add()** appended the new distribution instead of keeping `choices` sorted by `id`, as the class documents ([#49](https://github.com/brightway-lca/stats_arrays/issues/49)). `MCRandomNumberGenerator` relies on that order to match each distribution to its rows in the id-sorted params, so after adding a distribution with a lower id than one already present, distributions were sampled with other distributions' parameters. `add()` now keeps `choices` sorted.

### Performance

* Lognormal and normal samples are drawn with `Generator.standard_normal` and scaled, shifted and exponentiated in place. For 626,306 lognormal parameters (ecoinvent 3.13 scale), `MCRandomNumberGenerator.generate(out=...)` takes about 16 ms instead of about 50 ms in 4.0.
* `bounded_random_variables` returns early when no row has a `minimum` or `maximum`, instead of building bounds masks over all the samples. The check reads the current params on every draw.
* `generate()` writes each distribution block straight to its final positions, instead of filling a zeroed array and then copying the whole vector through `reverse_ordering`.
* `DiscreteUniform` draws all rows in one vectorized call with a `Generator`.

# 4.0 (2026-10-20)

### Breaking changes

* `UncertaintyBase.id` and `UncertaintyBase.description` are now declared but unset on the abstract base class, rather than defaulting to `None`. Every concrete distribution sets both, so `NormalUncertainty.id` is unchanged; only reading `id`/`description` directly off `UncertaintyBase` is affected, and that now raises `AttributeError` instead of returning `None`. This lets `id` be typed as `int` rather than `Optional[int]`, and lets `uncertainty_choices.add()` reject distributions without an integer `id` (see Bug fixes).
* The **Normal**, **Lognormal**, **Geometric**, **Beta** and **DiscreteUniform** modules no longer have a `stats` attribute, because `scipy.stats` is now imported on first use (see Performance). Code that reached `scipy.stats` through, for example, `stats_arrays.distributions.normal.stats` should import `scipy.stats` directly.

### New features

* `stats_arrays` is now a **typed library** (PEP 561). The package ships a `py.typed` marker, so type checkers use the inline annotations in downstream code instead of treating every import as `Any`. The whole package passes `mypy` with `disallow_untyped_defs` and `check_untyped_defs`, and that check runs in CI and as a pre-commit hook.
* Two public type aliases are now exported for annotating your own code:
  * `ParamsArray` — the structured NumPy array of uncertainty parameters used throughout the library.
  * `StatisticsResult` — a `TypedDict` for the return value of `statistics()`, with keys `mean`, `mode`, `median`, `lower` and `upper`, each `Optional[float]`. Misspelled keys are now a type error at check time.

### Bug fixes

* **UncertaintyChoices.add()** — the guard rejecting distributions without an integer `id` could never fire: `hasattr(distribution, "id")` was always true because the base class supplied `None`, and the two conditions were combined with `and` rather than `or`. It now checks `isinstance(getattr(distribution, "id", None), int)`.
* **UncertaintyChoices** — the duplicate-id `ValueError` and the already-present `UserWarning` both formatted a class object with `{:d}`, which would itself raise `TypeError` instead of reporting the actual problem.
* **UncertaintyBase.pdf()** — the base class declared a return type of `NDArray`, but every implementation returns a `(xs, ys)` tuple, as its own docstring describes. The annotation now says `Tuple[NDArray, NDArray]`.

### Performance

* `import stats_arrays` no longer imports `scipy.stats`, which took most of its import time (about 400 ms down to 45 ms). The `cdf`, `ppf` and `pdf` methods of **Normal**, **Lognormal**, **Geometric**, **Beta** and **DiscreteUniform** import it on first use.

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
