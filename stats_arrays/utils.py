from .errors import MultipleRowParamsArrayError
from functools import wraps
from numpy import zeros, NaN, ndarray


def flatten_numpy_array(obj):
    if not isinstance(obj, ndarray):
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
                      kwargs.iteritems()])
        return function(cls, params, *args, **kwargs)
    return wrapper


def construct_params_array(length=1, include_type=False):
    dtype = [('loc', 'f4'), ('scale', 'f4'),
             ('minimum', 'f4'), ('maximum', 'f4'),
             ('negative', 'b1')]
    if include_type:
        dtype.append(('uncertainty_type', 'u4'))
    params = zeros((length,), dtype=dtype)
    params['minimum'] = params['maximum'] = params['scale'] = NaN
    return params


def from_dicts(objs):
    params_array = construct_params_array(len(objs), include_type=True)
    for key in ("loc", "scale", "minimum", "maximum", "negative",
                "uncertainty_type"):
        params_array[key] = tuple([o.get(key, NaN) for o in objs])
    return params_array
