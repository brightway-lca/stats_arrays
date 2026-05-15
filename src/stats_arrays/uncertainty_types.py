from enum import IntEnum


class UncertaintyType(IntEnum):
    """Integer enum of all built-in uncertainty distribution IDs.

    Each member value equals the ``id`` attribute of the corresponding
    distribution class and the integer stored in a params array's
    ``uncertainty_type`` field, so all three forms are interchangeable::

        UncertaintyType.normal          # 3  (enum member)
        NormalUncertainty.id            # 3  (plain int)
        params["uncertainty_type"]      # 3  (numpy uint8)

        # All comparisons below are True
        UncertaintyType.normal == 3
        uncertainty_choices[UncertaintyType.normal] is NormalUncertainty
        params["uncertainty_type"] == UncertaintyType.normal
    """

    undefined = 0
    no_uncertainty = 1
    lognormal = 2
    normal = 3
    uniform = 4
    triangular = 5
    bernoulli = 6
    discrete_uniform = 7
    weibull = 8
    gamma = 9
    beta = 10
    generalized_extreme_value = 11
    students_t = 12
    beta_pert = 13
