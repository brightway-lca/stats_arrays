class StatsArraysError(Exception):
    """Base class for `stats_arrays` errors"""


class ImproperBoundsError(StatsArraysError):
    """The bounds and mean specified are illegal for this distribution."""


class MaximumIterationsError(StatsArraysError):
    """Drawing random numbers from the distribution to fit in the bounds specified used more than the maximum number of iterations allowed."""


class UnknownUncertaintyType(StatsArraysError):
    """The uncertainty type is not defined in uncertainty_choices."""


class UndefinedDistributionError(StatsArraysError):
    """Values were attempted to be calculated by an undefined distribution."""


class InvalidParamsError(StatsArraysError):
    """Invalid params array passed to uncertainty distribution init."""


class MultipleRowParamsArrayError(StatsArraysError):
    """A function or method which doesn't accept it was passed a params array with more than one row."""


class UnreasonableBoundsError(StatsArraysError):
    """The provided bounds cover an unreasonably small section of the distribution sample space."""
