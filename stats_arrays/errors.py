class StatsArraysError(Exception):
    """Base class for `stats_arrays` errors"""
    pass


class ImproperBoundsError(StatsArraysError):
    """The bounds and mean specified are illegal for this distribution."""
    pass


class MaximumIterationsError(StatsArraysError):
    """Drawing random numbers from the distribution to fit in the bounds specified used more than the maximum number of iterations allowed."""
    pass


class UnknownUncertaintyType(StatsArraysError):
    """The uncertainty type is not defined in uncertainty_choices."""
    pass


class UndefinedDistributionError(StatsArraysError):
    """Values were attempted to be calculated by an undefined distribution."""
    pass


class InvalidParamsError(StatsArraysError):
    """Invalid params array passed to uncertainty distribution init."""
    pass


class MultipleRowParamsArrayError(StatsArraysError):
    """A function or method which doesn't accept it was passed a params array with more than one row."""
    pass


class UnreasonableBoundsError(StatsArraysError):
    """The provided bounds cover an unreasonably small section of the distribution sample space."""
    pass
