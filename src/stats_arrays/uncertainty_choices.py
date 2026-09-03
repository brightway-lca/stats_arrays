import warnings
from collections.abc import Iterable as IterableABC
from typing import Iterator, Type, TypeVar

from stats_arrays.distributions import (
    BernoulliUncertainty,
    BetaUncertainty,
    BetaPERTUncertainty,
    DiscreteUniform,
    GammaUncertainty,
    GeneralizedExtremeValueUncertainty,
    LognormalUncertainty,
    NormalUncertainty,
    NoUncertainty,
    StudentsTUncertainty,
    TriangularUncertainty,
    UncertaintyBase,
    UndefinedUncertainty,
    UniformUncertainty,
    WeibullUncertainty,
)

DISTRIBUTIONS = (
    BernoulliUncertainty,
    BetaUncertainty,
    BetaPERTUncertainty,
    DiscreteUniform,
    GammaUncertainty,
    GeneralizedExtremeValueUncertainty,
    LognormalUncertainty,
    NormalUncertainty,
    NoUncertainty,
    StudentsTUncertainty,
    TriangularUncertainty,
    UndefinedUncertainty,
    UniformUncertainty,
    WeibullUncertainty,
)


DistributionType = TypeVar("DistributionType", bound=UncertaintyBase, covariant=True)


class UncertaintyChoices(IterableABC[Type[UncertaintyBase]]):
    """A container for uncertainty distributions, keyed by integer ID.

    Use :class:`stats_arrays.UncertaintyType` for named constants instead of
    raw integer IDs::

        uncertainty_choices[UncertaintyType.normal]   # NormalUncertainty
        uncertainty_choices[3]                        # same thing
    """

    def __init__(self):
        # Sorted by id
        self.choices: list = sorted(DISTRIBUTIONS, key=lambda x: x.id)
        self.check_id_uniqueness()

    def check_id_uniqueness(self) -> None:
        self.id_dict = {}
        for dist in self.choices:
            if dist.id in self.id_dict:
                raise ValueError(
                    "Uncertainty id {} is already in use by {}".format(
                        dist.id, self.id_dict[dist.id].__name__
                    )
                )
            self.id_dict[dist.id] = dist

    def __iter__(self) -> Iterator[Type[UncertaintyBase]]:
        return iter(self.choices)

    def __getitem__(self, id_: int) -> Type[UncertaintyBase]:
        return self.id_dict[id_]

    def __len__(self) -> int:
        return len(self.id_dict)

    def __contains__(self, choice: Type[UncertaintyBase]) -> bool:
        return choice in self.choices

    def add(self, distribution: Type[UncertaintyBase]) -> None:
        if not isinstance(getattr(distribution, "id", None), int):
            raise ValueError(
                "Uncertainty distributions must have integer `id` attribute."
            )
        if distribution.id in self.id_dict:
            warnings.warn(
                "ERROR: This distribution (id {}) is already present!".format(
                    distribution.id
                )
            )
            return
        self.choices.append(distribution)
        self.id_dict[distribution.id] = distribution


uncertainty_choices = UncertaintyChoices()
