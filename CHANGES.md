# stats_arrays Changelog

# 0.7 (2024-08-19)

* Moved to `pyproject.toml` packaging and `pytest` tests
* Numpy 2.0 compatiblity

## 0.6.6 (2023-10-18)

* Merged [PR #13](https://github.com/brightway-lca/stats_arrays/pull/13). Move argsort to save on execution time. Thanks @Loisel!
* Merged [PR #12](https://github.com/brightway-lca/stats_arrays/pull/12). Update links in readme. Thanks @mfastudillo!
* Merged [PR #10](https://github.com/brightway-lca/stats_arrays/pull/10). Fix flaky test on distributions::extreme.py, distributions::gama.py, and distributions::student.py. Thanks @lonly7star!

## 0.6.5 (2021-05-06)

* Updates for changes in Numpy API

## 0.6.4 (2020-01-31)

* Merged [PR #5](https://bitbucket.org/cmutel/stats_arrays/pull-requests/5/use-meanround-0-instead-of-round-mean/diff), fix rounding function.

## 0.6.3 (2019-11-29)

* Improve speed of lognormal CDF and PPF under common conditions

## 0.6.2 (2019-11-13)

* Restore Python 2.7 compatibility

## 0.6.1 (2019-07-29)

* Several bug fixes from Daniel de Koning
* Start improvement of beta distribution

# 0.6 (2019-05-19)

* Allow `MCRandomNumberGenerator` to generate multiple samples at once
* Move tests to pytest
* Add tests for MCRNG

## 0.5.1 (2019-05-11)

* Import from `collections.abc` when possible, for Python 3.8 support.
