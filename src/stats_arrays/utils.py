from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from stats_arrays.errors import MultipleRowParamsArrayError

BASE_DTYPE_FIELDS: List[Tuple[str, type]] = [
    ("loc", np.float64),
    ("scale", np.float64),
    ("shape", np.float64),
    ("minimum", np.float64),
    ("maximum", np.float64),
    ("negative", bool),
]
BASE_DTYPE: np.dtype = np.dtype(BASE_DTYPE_FIELDS)

ParamsArray = npt.NDArray[np.void]
"""A :ref:`params-array`: a structured NumPy array whose fields are ``loc``, ``scale``,
``shape``, ``minimum``, ``maximum``, ``negative``, and — for a :ref:`hpa` — the
``uncertainty_type`` distribution id.

Structured arrays all share the ``numpy.void`` scalar type, so this alias documents
intent rather than pinning an exact dtype; the field layout is not expressible in the
NumPy typing system today. Field access (``params["loc"]``) is therefore untyped.
"""


class StatisticsResult(TypedDict):
    """Summary statistics returned by :meth:`UncertaintyBase.statistics`.

    Every key is always present. A statistic that is undefined for a given
    distribution and parameter set is ``None`` rather than omitted.
    """

    mean: Optional[float]
    mode: Optional[float]
    median: Optional[float]
    lower: Optional[float]
    upper: Optional[float]


RandomSource = Union[np.random.RandomState, np.random.Generator]
"""A NumPy source of random numbers: either the legacy ``numpy.random.RandomState``
or a ``numpy.random.Generator``."""

SeedLike = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
    np.random.RandomState,
]
"""Anything accepted as the ``seed`` argument of the random number generator classes."""


def make_random_source(seed: SeedLike = None) -> RandomSource:
    """Turn the ``seed`` argument into a source of random numbers.

    * ``None``: an unseeded ``numpy.random.Generator``.
    * An integer, sequence of integers, ``SeedSequence`` or ``BitGenerator``:
      ``numpy.random.default_rng(seed)``.
    * A ``Generator`` or ``RandomState`` instance: used as is (not copied).

    To get the legacy (stats_arrays < 5) stream for an integer seed, pass
    ``numpy.random.RandomState(seed)``.
    """
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    if isinstance(seed, bool):
        raise TypeError("`seed` must not be a boolean")
    return np.random.default_rng(seed)


def random_integers(
    random: RandomSource, low: Any, high: Any = None, size: Any = None
) -> npt.NDArray:
    """Draw integers on ``[low, high)`` from either kind of random source.

    ``RandomState.randint`` and ``Generator.integers`` both exclude ``high`` by default.
    """
    if isinstance(random, np.random.Generator):
        return random.integers(low, high, size=size)
    return random.randint(low, high, size=size)


AnyCallable = TypeVar("AnyCallable", bound=Callable[..., Any])


@overload
def flatten_numpy_array(obj: npt.NDArray) -> npt.NDArray: ...


@overload
def flatten_numpy_array(obj: Any) -> Any: ...


def flatten_numpy_array(obj: Any) -> Any:
    if not isinstance(obj, np.ndarray):
        return obj
    return obj.ravel()


def one_row_params_array(function: AnyCallable) -> AnyCallable:
    """Reshape ``params`` to a single row, and flatten any other array arguments.

    Raises ``stats_arrays.MultipleRowParamsArrayError`` if ``params`` has more than one
    row. The wrapped callable keeps the signature of ``function``.
    """

    @wraps(function)
    def wrapper(cls: Any, params: ParamsArray, *args: Any, **kwargs: Any) -> Any:
        if len(params.shape) == 1:
            params = params.reshape(params.shape[0], 1)
        else:
            if params.shape[0] != 1:
                raise MultipleRowParamsArrayError
        # Flatten any additional inputs to one dimension
        # Needed for PDF optional xs input
        flattened_args = [flatten_numpy_array(x) for x in args]
        flattened_kwargs = {
            key: flatten_numpy_array(obj) for key, obj in kwargs.items()
        }
        return function(cls, params, *flattened_args, **flattened_kwargs)

    return cast(AnyCallable, wrapper)


def construct_params_array(length: int = 1, include_type: bool = False) -> ParamsArray:
    dtype: np.dtype
    if include_type:
        dtype = np.dtype(BASE_DTYPE_FIELDS + [("uncertainty_type", np.uint8)])
    else:
        dtype = BASE_DTYPE
    params = np.zeros((length,), dtype=dtype)
    params["minimum"] = params["maximum"] = np.nan
    params["scale"] = params["loc"] = params["shape"] = np.nan
    return params


def rescale_to_unitary_interval(params: ParamsArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Rescale params to a (0,1) interval. Return adjusted `loc` and scale (`minimum - maximum`).

    Uses default values of (0, 1) for minimum and maximum if not present.

    Needed because SciPy assumes a (0,1) interval for many distributions."""
    minimum = params["minimum"].copy()
    maximum = params["maximum"].copy()

    minimum[np.isnan(minimum)] = 0
    maximum[np.isnan(maximum)] = 1

    scale = maximum - minimum
    adjusted_loc = (params["loc"] - minimum) / scale
    return adjusted_loc, scale


def rescale_vector_to_params(params: npt.NDArray, vector: npt.NDArray) -> npt.NDArray:
    """Unscale `vector` from a (0,1) interval to the `(params["maximum"] - params["minimum"])`."""
    minimum = params["minimum"].copy()
    maximum = params["maximum"].copy()

    # Handle NaN values by defaulting to (0, 1)
    minimum[np.isnan(minimum)] = 0
    maximum[np.isnan(maximum)] = 1

    scale = maximum - minimum

    # Handle broadcasting for multiple rows
    if vector.ndim == 2 and scale.ndim == 1:
        # vector shape: (n_rows, n_samples), scale/minimum shape: (n_rows,)
        return vector * scale[:, np.newaxis] + minimum[:, np.newaxis]
    else:
        # Single row case or matching dimensions
        return vector * scale + minimum
