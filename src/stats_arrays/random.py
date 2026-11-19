from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import UncertaintyBase
from stats_arrays.distributions.undefined import UndefinedUncertainty
from stats_arrays.errors import UnknownUncertaintyType, UnreasonableBoundsError
from stats_arrays.uncertainty_choices import uncertainty_choices
from stats_arrays.utils import (
    ParamsArray,
    RandomSource,
    SeedLike,
    make_random_source,
    random_integers,
)


class RandomNumberGenerator(Iterable[npt.NDArray]):
    def __init__(
        self,
        uncertainty_type: Type[UncertaintyBase],
        params: ParamsArray,
        size: int = 1,
        maximum_iterations: int = 100,
        seed: SeedLike = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a random number generator from a :ref:`params-array` and an uncertainty distribution.

        Upon instantiation, the class checks that:

        * The ``minimum`` and ``maximum`` bounds, if any, are reasonable
        * The given uncertainty type can be used

        ``uncertainty_type`` is a subclass of :ref:`UncertaintyBase`.

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
            * *seed* (optional): Seed for the random number generator. An integer, sequence of integers, ``SeedSequence`` or ``BitGenerator`` is passed to ``numpy.random.default_rng``; a ``numpy.random.Generator`` or ``numpy.random.RandomState`` instance is used as is. Default is ``None`` (unseeded ``Generator``). To reproduce results from stats_arrays < 5, pass ``numpy.random.RandomState(seed)``.

        Returns:
            A class instance

        """
        self.params = params
        self.length = self.params.shape[0]
        self.size = size
        self.uncertainty_type = uncertainty_type
        self.maximum_iterations = maximum_iterations
        self.random: RandomSource = make_random_source(seed)
        self.verify_uncertainty_type()
        self.verify_params()

    def verify_params(
        self,
        params: Optional[ParamsArray] = None,
        uncertainty_type: Optional[Type[UncertaintyBase]] = None,
    ) -> None:
        """Verify that parameters are within bounds. Mean is not restricted to bounds, unless the distribution requires it (e.g. triangular)."""
        if params is None:  # Can't convert array to boolean
            params = self.params
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        uncertainty_type.validate(params)

    def verify_uncertainty_type(
        self, uncertainty_type: Optional[Type[UncertaintyBase]] = None
    ) -> None:
        """Make sure the given uncertainty type provides the method ``bounded_random_variables``."""
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if not hasattr(uncertainty_type, "bounded_random_variables"):
            raise UnknownUncertaintyType(
                "The provided uncertainty type must have the "
                "`bounded_random_variables` method."
            )

    def generate_random_numbers(
        self,
        uncertainty_type: Optional[Type[UncertaintyBase]] = None,
        params: Optional[ParamsArray] = None,
        size: Optional[int] = None,
    ) -> npt.NDArray:
        if not uncertainty_type:
            uncertainty_type = self.uncertainty_type
        if params is None:  # Can't convert array to boolean
            params = self.params
        if not size:
            size = self.size
        return uncertainty_type.bounded_random_variables(
            params, size, self.random, self.maximum_iterations
        )

    def next(self) -> npt.NDArray:
        return self.generate_random_numbers()

    def __next__(self) -> npt.NDArray:
        return self.next()

    def __iter__(self) -> Iterator[npt.NDArray]:
        return self


class MCRandomNumberGenerator(Iterable[npt.NDArray]):
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
        * *seed* (optional): Seed for the random number generator. An integer, sequence of integers, ``SeedSequence`` or ``BitGenerator`` is passed to ``numpy.random.default_rng``; a ``numpy.random.Generator`` or ``numpy.random.RandomState`` instance is used as is. Default is ``None`` (unseeded ``Generator``). To reproduce results from stats_arrays < 5, pass ``numpy.random.RandomState(seed)``.

    Returns:
        A class instance

    Internally, the parameters are sorted by ``uncertainty_type``; ``ordering`` is the
    permutation used, so that ``params[ordering]`` is in sorted order. See
    :meth:`generate` for how to get values in that order directly.

    """

    def __init__(
        self,
        params: ParamsArray,
        maximum_iterations: int = 50,
        seed: SeedLike = None,
        **kwargs: Any,
    ) -> None:
        self.params = params.copy()
        self.length = self.params.shape[0]
        self.maximum_iterations = maximum_iterations
        self.choices = uncertainty_choices
        self.random: RandomSource = make_random_source(seed)
        self.verify_params()
        # Default (non-stable) sort kept deliberately: with a stable sort,
        # ``seed=RandomState(n)`` would no longer reproduce stats_arrays < 5.
        self.ordering = np.argsort(self.params["uncertainty_type"])
        self.reverse_ordering = np.argsort(self.ordering)
        self._identity_ordering = bool((self.ordering == np.arange(self.length)).all())
        self.params = self.params[self.ordering]
        self.positions = self.get_positions()
        self._blocks = self._build_blocks()

    def get_positions(self) -> Dict[Type[UncertaintyBase], int]:
        """Construct dictionary of where each distribution starts and stops in the sorted parameter array"""
        return dict(
            [
                (choice, (self.params["uncertainty_type"] == choice.id).sum())
                for choice in self.choices
            ]
        )

    def _build_blocks(self) -> "list[tuple[Type[UncertaintyBase], int, int]]":
        """Precompute ``(uncertainty_type, start, stop)`` for each distribution
        present in the sorted parameter array."""
        blocks = []
        offset = 0
        for uncertainty_type in self.choices:
            numparams = int(self.positions[uncertainty_type])
            if not numparams:
                continue
            blocks.append((uncertainty_type, offset, offset + numparams))
            offset += numparams
        return blocks

    def verify_params(self) -> None:
        """Verify that all uncertainty types are allowed, and parameter validate using distribution class methods"""
        ids = set(np.unique(self.params["uncertainty_type"]))
        extra_ids = ids.difference(set([x.id for x in self.choices]))
        if extra_ids:
            raise ValueError(
                "Uncertainty type id(s) {} are not valid".format(extra_ids)
            )

        for uncertainty_type in self.choices:
            mask = self.params["uncertainty_type"] == uncertainty_type.id
            if mask.sum():
                uncertainty_type.validate(self.params[mask])

    def generate(
        self,
        samples: int = 1,
        out: Optional[npt.NDArray] = None,
        order: Literal["original", "sorted"] = "original",
    ) -> npt.NDArray:
        """Generate random samples.

        If ``samples`` is one, return a one-dimensional array. Otherwise returns a ``num_parameters, samples`` array.

        Args:
            * *samples* (int, optional): Number of samples to draw for each parameter. Default is ``1``.
            * *out* (array, optional): A preallocated float array of shape ``(num_parameters,)`` (when ``samples`` is one) or ``(num_parameters, samples)`` to write the samples into, avoiding an allocation per call. It is returned.
            * *order* (str, optional): ``"original"`` (default) returns values in the row order of the ``params`` given at instantiation. ``"sorted"`` returns them in the internal order, sorted by ``uncertainty_type``: row ``i`` of the result belongs to row ``ordering[i]`` of the original ``params``. This skips a scatter over the whole vector, so callers who keep their own index arrays can apply ``ordering`` to them once and then use this order on every draw.

        Returns:
            The array of samples.
        """
        if order not in ("original", "sorted"):
            raise ValueError(f"`order` must be 'original' or 'sorted', got {order!r}")
        shape = (self.length,) if samples == 1 else (self.length, samples)
        if out is None:
            out = np.empty(shape)
        elif not isinstance(out, np.ndarray) or not np.issubdtype(
            out.dtype, np.floating
        ):
            kind = (
                f"{out.dtype} array"
                if isinstance(out, np.ndarray)
                else type(out).__name__
            )
            raise TypeError(f"`out` must be a float NumPy array, got {kind}")
        elif out.shape != shape:
            raise ValueError(f"`out` must have shape {shape}, got {out.shape}")

        # In original order, scatter each block straight to its final positions
        # instead of filling a sorted array and then permuting the whole thing.
        destinations: Optional[npt.NDArray] = (
            self.ordering
            if order == "original" and not self._identity_ordering
            else None
        )

        for uncertainty_type, start, stop in self._blocks:
            # Always go through ``bounded_random_variables``: it reads the
            # current bounds on every draw (and returns early when there are
            # none), so changes to ``self.params`` after instantiation count.
            random_data = uncertainty_type.bounded_random_variables(
                self.params[start:stop],
                samples,
                self.random,
                self.maximum_iterations,
            )
            if samples == 1 and random_data.ndim == 2:
                random_data = random_data[:, 0]  # Restore to 1-d
            if destinations is None:
                out[start:stop] = random_data
            else:
                out[destinations[start:stop]] = random_data

        self.random_data = out
        return out

    def next(self) -> npt.NDArray:
        """Generate a new vector of random numbers"""
        return self.generate()

    def __next__(self) -> npt.NDArray:
        return self.next()

    def __iter__(self) -> Iterator[npt.NDArray]:
        return self


class LatinHypercubeRNG(MCRandomNumberGenerator):
    """
    A random number generator that pre-calculates a sample space to draw from.

    .. rubric:: Inputs

    * params : A :ref:`params-array` which gives parameters for distributions (one distribution per row).
    * seed : Optional. An integer, sequence of integers, ``SeedSequence`` or ``BitGenerator`` passed to ``numpy.random.default_rng``, or a ``numpy.random.Generator`` or ``numpy.random.RandomState`` instance used as is. To reproduce results from stats_arrays < 5, pass ``numpy.random.RandomState(seed)``.
    * samples : An integer number of samples to construct for each distribution.
    """

    def __init__(
        self,
        params: ParamsArray,
        seed: SeedLike = None,
        samples: int = 10,
        **kwargs: Any,
    ) -> None:
        self.params = params
        self.length = self.params.shape[0]
        self.row_index = np.arange(self.length)
        self.size = 1
        self.samples = samples
        self.choices = uncertainty_choices
        self.random = make_random_source(seed)
        self.uncertainty_type = None

        self.verify_params()
        self.build_hypercube()

    def build_hypercube(self) -> None:
        """Build an array, of shape `self.length` rows by `self.samples` columns, which contains the sample space to be drawn from when doing Latin Hypercubic sampling.

        Each row represents a different data point and distribution. The final sample space is `self.hypercube`. A distribution is usable if it implements ``ppf``, and, for rows with ``minimum`` or ``maximum`` bounds, ``cdf``; otherwise ``NotImplementedError`` is raised. Of the built-in distributions, Weibull, gamma, generalized extreme value and Student's t implement neither.

        .. rubric:: Builds

        self.hypercube : Numpy array with dimensions `self.length` by `self.samples`."""
        self._check_distributions_supported()
        # The range of percentages to sample from for each row: the whole
        # distribution by default, or the part inside its bounds.
        lower = np.zeros(self.length)
        upper = np.ones(self.length)
        for uncertainty_type in self.choices:
            mask = self.params["uncertainty_type"] == uncertainty_type.id
            if not mask.any() or issubclass(uncertainty_type, UndefinedUncertainty):
                # Undefined distributions return ``loc`` whatever the
                # percentages, and have no ``cdf``.
                continue
            lower[mask], upper[mask] = self._percentage_bounds(
                uncertainty_type, self.params[mask]
            )
        # ``samples`` evenly spaced percentages strictly inside each range.
        fractions = np.arange(1, self.samples + 1) / (self.samples + 1)
        self.percentages = lower.reshape(-1, 1) + (upper - lower).reshape(
            -1, 1
        ) * fractions.reshape(1, -1)
        # Bounds that hold none of the distribution, or so little that it is
        # lost to rounding, leave nothing to sample: every percentage would be
        # 0 or 1, and ``ppf`` would return infinities or the edge of the support.
        empty = (
            ~(upper > lower)
            | (self.percentages.min(axis=1) <= 0)
            | (self.percentages.max(axis=1) >= 1)
        )
        if empty.any():
            raise UnreasonableBoundsError(
                "The `minimum` and `maximum` bounds contain no part of the "
                f"distribution that can be sampled. {UncertaintyBase._fmt_bad_rows(empty)}"
            )
        # Transform percentages into a sample space
        self.hypercube = np.zeros((self.length, self.samples))
        for uncertainty_type in self.choices:
            mask = self.params["uncertainty_type"] == uncertainty_type.id
            if not mask.sum():
                continue
            self.hypercube[mask, :] = uncertainty_type.ppf(
                params=self.params[mask], percentages=self.percentages[mask, :]
            )
        self.hypercube[self.params["negative"], :] = (
            -1 * self.hypercube[self.params["negative"], :]
        )

    def _check_distributions_supported(self) -> None:
        """Raise ``NotImplementedError`` naming the distributions in ``params``
        that lack the ``ppf`` (or, for bounded rows, ``cdf``) needed here."""

        def implemented(uncertainty_type: Type[UncertaintyBase], name: str) -> bool:
            method = getattr(uncertainty_type, name)
            base = getattr(UncertaintyBase, name)
            return getattr(method, "__func__", method) is not base.__func__

        problems = []
        unsupported = np.zeros(self.length, dtype=bool)
        for uncertainty_type in self.choices:
            mask = self.params["uncertainty_type"] == uncertainty_type.id
            if not mask.any() or issubclass(uncertainty_type, UndefinedUncertainty):
                continue
            missing = []
            if not implemented(uncertainty_type, "ppf"):
                missing.append("ppf")
            bounded = mask & ~(
                np.isnan(self.params["minimum"]) & np.isnan(self.params["maximum"])
            )
            if bounded.any() and not implemented(uncertainty_type, "cdf"):
                missing.append("cdf")
                if "ppf" not in missing:
                    mask = bounded
            if missing:
                problems.append(
                    f"{uncertainty_type.__name__} (id {uncertainty_type.id}) "
                    f"does not implement {' or '.join(f'`{m}`' for m in missing)}"
                )
                unsupported |= mask
        if problems:
            raise NotImplementedError(
                "LatinHypercubeRNG needs `ppf` to sample a distribution, and `cdf` "
                f"to apply bounds: {'; '.join(problems)}. "
                f"{UncertaintyBase._fmt_bad_rows(unsupported)}"
            )

    @staticmethod
    def _percentage_bounds(
        uncertainty_type: Type[UncertaintyBase], params: ParamsArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Return the range of percentages, ``(lower, upper)``, that ``ppf`` maps
        inside the ``minimum`` and ``maximum`` bounds of each row in ``params``.

        Rows without bounds get ``(0, 1)``. A lower bound starts at
        P(X < minimum) rather than P(X <= minimum), so that a discrete
        distribution can still return ``minimum`` itself. Distributions whose
        ``ppf`` decreases (a lognormal with ``negative`` set) have the roles of
        the two bounds swapped.
        """
        length = params.shape[0]
        lower = np.zeros(length)
        upper = np.ones(length)
        has_min = ~np.isnan(params["minimum"])
        has_max = ~np.isnan(params["maximum"])
        if not (has_min.any() or has_max.any()):
            return lower, upper

        def cdf(rows: npt.NDArray, values: npt.NDArray) -> npt.NDArray:
            # ``cdf`` may change ``values`` in place, so always pass a new array
            return np.asarray(
                uncertainty_type.cdf(params[rows], np.array(values, dtype=float))
            ).reshape(-1)

        bounded = has_min | has_max
        quartiles = np.asarray(
            uncertainty_type.ppf(
                params=params[bounded],
                percentages=np.tile(np.array([0.25, 0.75]), (int(bounded.sum()), 1)),
            )
        )
        decreasing = np.zeros(length, dtype=bool)
        decreasing[bounded] = quartiles[:, 0] > quartiles[:, 1]

        if has_min.any():
            p = cdf(has_min, np.nextafter(params["minimum"][has_min], -np.inf))
            lower[has_min & ~decreasing] = p[~decreasing[has_min]]
            upper[has_min & decreasing] = p[decreasing[has_min]]
        if has_max.any():
            p = cdf(has_max, params["maximum"][has_max])
            upper[has_max & ~decreasing] = p[~decreasing[has_max]]
            lower[has_max & decreasing] = p[decreasing[has_max]]
        return lower, upper

    def next(self) -> npt.NDArray:
        """Draw directly from pre-computed sample space."""
        return self.hypercube[
            self.row_index, random_integers(self.random, self.samples, size=self.length)
        ]
