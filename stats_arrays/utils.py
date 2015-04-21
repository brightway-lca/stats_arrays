from .errors import MultipleRowParamsArrayError
from functools import wraps
import numpy as np

BASE_DTYPE = [
    (u'loc', np.float64),
    (u'scale', np.float64),
    (u'shape', np.float64),
    (u'minimum', np.float64),
    (u'maximum', np.float64),
    (u'negative', np.bool)
]


def flatten_numpy_array(obj):
    if not isinstance(obj, np.ndarray):
        return obj
    else:
        return obj.ravel()


def one_row_params_array(function):
    @wraps(function)
    def wrapper(cls, params, *args, **kwargs):
        if len(params.shape) == 1:
            params = params.reshape(params.shape[0], 1)
        else:
            if params.shape[0] != 1:
                raise MultipleRowParamsArrayError
        # Flatten any additional inputs to one dimension
        # Needed for PDF optional xs input
        args = [flatten_numpy_array(x) for x in args]
        kwargs = dict([(key, flatten_numpy_array(obj)) for key, obj in
                      kwargs.items()])
        return function(cls, params, *args, **kwargs)
    return wrapper


def construct_params_array(length=1, include_type=False):
    dtype = BASE_DTYPE
    if include_type:
        dtype = dtype + [(u'uncertainty_type', np.uint8)]
    params = np.zeros((length,), dtype=dtype)
    params[u'minimum'] = params[u'maximum'] = np.NaN
    params[u'scale'] = params[u'loc'] = params[u'shape'] = np.NaN
    return params
