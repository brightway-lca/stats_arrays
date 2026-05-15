from stats_arrays import (
    NormalUncertainty,
    LognormalUncertainty,
    UniformUncertainty,
    TriangularUncertainty,
    BernoulliUncertainty,
    DiscreteUniform,
    WeibullUncertainty,
    GammaUncertainty,
    BetaUncertainty,
    GeneralizedExtremeValueUncertainty,
    StudentsTUncertainty,
    BetaPERTUncertainty,
    NoUncertainty,
    UndefinedUncertainty,
    UncertaintyType,
    uncertainty_choices,
)


DISTRIBUTION_IDS = [
    (UncertaintyType.undefined,               UndefinedUncertainty),
    (UncertaintyType.no_uncertainty,          NoUncertainty),
    (UncertaintyType.lognormal,               LognormalUncertainty),
    (UncertaintyType.normal,                  NormalUncertainty),
    (UncertaintyType.uniform,                 UniformUncertainty),
    (UncertaintyType.triangular,              TriangularUncertainty),
    (UncertaintyType.bernoulli,               BernoulliUncertainty),
    (UncertaintyType.discrete_uniform,        DiscreteUniform),
    (UncertaintyType.weibull,                 WeibullUncertainty),
    (UncertaintyType.gamma,                   GammaUncertainty),
    (UncertaintyType.beta,                    BetaUncertainty),
    (UncertaintyType.generalized_extreme_value, GeneralizedExtremeValueUncertainty),
    (UncertaintyType.students_t,              StudentsTUncertainty),
    (UncertaintyType.beta_pert,               BetaPERTUncertainty),
]


def test_enum_members_equal_distribution_ids():
    """Each enum member value must match the corresponding class .id."""
    for member, cls in DISTRIBUTION_IDS:
        assert member == cls.id, f"{member.name} value {int(member)} != {cls.__name__}.id {cls.id}"


def test_enum_members_are_ints():
    """IntEnum members must compare equal to plain ints."""
    assert UncertaintyType.normal == 3
    assert UncertaintyType.lognormal == 2
    assert UncertaintyType.beta_pert == 13


def test_uncertainty_choices_lookup_by_enum():
    """uncertainty_choices[enum_member] must return the correct class."""
    for member, cls in DISTRIBUTION_IDS:
        assert uncertainty_choices[member] is cls


def test_numpy_comparison():
    """Enum members must work in numpy array comparisons."""
    params = NormalUncertainty.from_dicts(
        {"loc": 1, "scale": 0.5, "uncertainty_type": 3},
        {"loc": 2, "scale": 0.3, "uncertainty_type": 2},
    )
    mask = params["uncertainty_type"] == UncertaintyType.normal
    assert mask.tolist() == [True, False]


def test_all_distributions_covered():
    """Every distribution registered in uncertainty_choices has an enum member."""
    enum_values = {int(m) for m in UncertaintyType}
    for dist in uncertainty_choices:
        assert dist.id in enum_values, (
            f"{dist.__name__} (id={dist.id}) has no UncertaintyType member"
        )
