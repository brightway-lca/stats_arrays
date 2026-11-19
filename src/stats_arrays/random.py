from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Type

import numpy as np
import numpy.typing as npt

from stats_arrays.distributions.base import BoundedUncertaintyBase, UncertaintyBase
from stats_arrays.errors import UnknownUncertaintyType
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

    def _build_blocks(
        self,
    ) -> "list[tuple[Type[UncertaintyBase], int, int, bool]]":
        """Precompute ``(uncertainty_type, start, stop, skip_bounds)`` for each
        distribution present in the sorted parameter array.

        ``skip_bounds`` is ``True`` when no row of that block sets ``minimum`` or
        ``maximum`` and the distribution uses a stock ``bounded_random_variables``,
        so ``generate`` can call ``random_variables`` directly. Distributions that
        override ``bounded_random_variables`` always go through it.
        """
        stock = {
            UncertaintyBase.bounded_random_variables.__func__,  # type: ignore[attr-defined]
            BoundedUncertaintyBase.bounded_random_variables.__func__,  # type: ignore[attr-defined]
        }
        blocks = []
        offset = 0
        for uncertainty_type in self.choices:
            numparams = int(self.positions[uncertainty_type])
            if not numparams:
                continue
            block = self.params[offset : offset + numparams]
            blocks.append(
                (
                    uncertainty_type,
                    offset,
                    offset + numparams,
                    uncertainty_type.bounded_random_variables.__func__ in stock  # type: ignore[attr-defined]
                    and not uncertainty_type.has_bounds(block),
                )
            )
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
        elif out.shape != shape:
            raise ValueError(f"`out` must have shape {shape}, got {out.shape}")

        # In original order, scatter each block straight to its final positions
        # instead of filling a sorted array and then permuting the whole thing.
        destinations: Optional[npt.NDArray] = (
            self.ordering
            if order == "original" and not self._identity_ordering
            else None
        )

        for uncertainty_type, start, stop, skip_bounds in self._blocks:
            params = self.params[start:stop]
            if skip_bounds:
                random_data = uncertainty_type.random_variables(
                    params, samples, self.random
                )
            else:
                random_data = uncertainty_type.bounded_random_variables(
                    params,
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

        Each row represents a different data point and distribution. The final sample space is `self.hypercube`. All distributions from `uncertainty_choices` are usable, and bounded distributions are also fine.

        .. rubric:: Builds

        self.hypercube : Numpy array with dimensions `self.length` by `self.samples`."""
        step_size = 1 / (self.samples + 1)
        # Define the beginning points and step sizes - not all the same because
        # some distributions are bounded. Make adjustments to generic values
        # later.
        inputs = np.tile(np.array((step_size, step_size)), self.length).reshape(
            self.length, 2
        )
        for uncertainty_type in self.choices:
            mask = self.params["uncertainty_type"] == uncertainty_type.id
            if not mask.sum():
                continue
            subarray = self.params[mask]
            # Adjust inputs when bounds are present. Easiest to do in three
            # discrete steps. First, when only a lower bound is present.
            only_min_mask = ~np.isnan(subarray["minimum"]) * np.isnan(
                subarray["maximum"]
            )
            if only_min_mask.sum():
                mins = uncertainty_type.cdf(
                    subarray[only_min_mask], subarray[only_min_mask]["minimum"]
                )
                steps = (1 - mins) / (self.samples + 1)
                inputs[mask, :][only_min_mask] = np.hstack((mins + steps, steps))
            # Next, if only a max bound is present
            only_max_mask = np.isnan(subarray["minimum"]) * ~np.isnan(
                subarray["maximum"]
            )
            if only_max_mask.sum():
                maxs = uncertainty_type.cdf(
                    subarray[only_max_mask], subarray[only_max_mask]["maximum"]
                )
                steps = maxs / (self.samples + 1)
                inputs[mask, :][only_max_mask] = np.hstack((steps, steps))
            # Finally, if both min and max bounds are present
            both_mask = ~np.isnan(subarray["minimum"]) * ~np.isnan(subarray["maximum"])
            if both_mask.sum():
                mins = uncertainty_type.cdf(
                    subarray[both_mask], subarray[both_mask]["minimum"]
                )
                maxs = uncertainty_type.cdf(
                    subarray[both_mask], subarray[both_mask]["maximum"]
                )
                steps = (maxs - mins) / (self.samples + 1)
                inputs[mask, :][both_mask] = np.hstack((mins + steps, steps))
        # Percentages is now a list, samples wide, of the percentages covered
        # by the bounded or unbounded distributions.
        self.percentages = inputs[:, 0].reshape(self.length, 1) + np.arange(
            0, self.samples, 1
        ) * inputs[:, 1].reshape(self.length, 1)
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

    def next(self) -> npt.NDArray:
        """Draw directly from pre-computed sample space."""
        return self.hypercube[
            self.row_index, random_integers(self.random, self.samples, size=self.length)
        ]
