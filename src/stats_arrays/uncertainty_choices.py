import warnings
from collections.abc import Iterable as IterableABC
from typing import Iterator, Type, TypeVar

from stats_arrays.distributions import (
    BernoulliUncertainty,
    BetaUncertainty,
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
    """An container for uncertainty distributions."""

    def __init__(self):
        # Sorted by id
        self.choices: list = sorted(DISTRIBUTIONS, key=lambda x: x.id)
        self.check_id_uniqueness()

    def check_id_uniqueness(self) -> None:
        self.id_dict = {}
        for dist in self.choices:
            if dist.id in self.id_dict:
                raise ValueError(
                    "Uncertainty id {:d} is already in use by {:d}".format(
                        dist.id, self.id_dict[dist.id]
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
        if not hasattr(distribution, "id") and isinstance(distribution.id, int):
            raise ValueError(
                "Uncertainty distributions must have integer `id` attribute."
            )
        if distribution.id in self.id_dict:
            warnings.warn(
                "ERROR: This distribution (id {:d}) is already present!".format(
                    distribution.id
                )
            )
            return
        self.choices.append(distribution)
        self.id_dict[distribution.id] = distribution


uncertainty_choices = UncertaintyChoices()
