class ImproperBoundsError(StandardError):
    """The bounds and mean specified are illegal for this distribution."""
    pass


class MaximumIterationsError(StandardError):
    """Drawing random numbers from the distribution to fit in the bounds specified used more than the maximum number of iterations allowed."""
    pass


class UnknownUncertaintyType(StandardError):
    """The uncertainty type is not defined in uncertainty_choices."""
    pass


class UndefinedDistributionError(StandardError):
    """Values were attempted to be calculated by an undefined distribution."""
    pass


class InvalidParamsError(StandardError):
    """Invalid params array passed to uncertainty distribution init."""
    pass


class MultipleRowParamsArrayError(StandardError):
    """A function or method which doesn't accept it was passed a params array with more than one row."""
    pass


class UnreasonableBoundsError(StandardError):
    """The provided bounds cover an unreasonably small section of the distribution sample space."""
    pass
