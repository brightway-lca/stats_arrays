from functools import wraps
from typing import Any, Callable, List, TypeVar, overload

import numpy as np
import numpy.typing as npt

from stats_arrays.errors import MultipleRowParamsArrayError

BASE_DTYPE_FIELDS: List[tuple] = [
    ("loc", np.float64),
    ("scale", np.float64),
    ("shape", np.float64),
    ("minimum", np.float64),
    ("maximum", np.float64),
    ("negative", bool),
]
BASE_DTYPE: npt.DTypeLike = np.dtype(BASE_DTYPE_FIELDS)

# Numpy typing isn't great, use this basic approach for now.
# Also consider nptyping: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
# TODO: Decide if we need separate types for when "uncertainty_type is included"
# @overload
# def construct_params_array(length: int = 1, include_type: bool = False) -> BaseParamsArray:
#     ...
# @overload
# def construct_params_array(length: int = 1, include_type: bool = True) -> ExtendedParamsArray:
#     ...
# More specific types for different parameter array structures
# BaseParamsArray = npt.NDArray[BASE_DTYPE_T]  # Without uncertainty_type
# ExtendedParamsArray = npt.NDArray[BASE_DTYPE_T]  # With uncertainty_type
BASE_DTYPE_T = TypeVar("BASE_DTYPE_T", bound=np.generic)
ParamsArray = npt.NDArray[BASE_DTYPE_T]



@overload
def flatten_numpy_array(obj: npt.NDArray) -> npt.NDArray:
    ...


@overload
def flatten_numpy_array(obj: Any) -> Any:
    ...


def flatten_numpy_array(obj):
    if not isinstance(obj, np.ndarray):
        return obj
    return obj.ravel()


def one_row_params_array(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(cls, params: ParamsArray, *args, **kwargs) -> Callable:
        if len(params.shape) == 1:
            params = params.reshape(params.shape[0], 1)
        else:
            if params.shape[0] != 1:
                raise MultipleRowParamsArrayError
        # Flatten any additional inputs to one dimension
        # Needed for PDF optional xs input
        args = [flatten_numpy_array(x) for x in args]
        kwargs = {key: flatten_numpy_array(obj) for key, obj in kwargs.items()}
        return function(cls, params, *args, **kwargs)

    return wrapper


def construct_params_array(length: int = 1, include_type: bool = False) -> ParamsArray:
    if include_type:
        dtype = np.dtype(BASE_DTYPE_FIELDS + [("uncertainty_type", np.uint8)])
    else:
        dtype = BASE_DTYPE
    params = np.zeros((length,), dtype=dtype)
    params["minimum"] = params["maximum"] = np.nan
    params["scale"] = params["loc"] = params["shape"] = np.nan
    return params
