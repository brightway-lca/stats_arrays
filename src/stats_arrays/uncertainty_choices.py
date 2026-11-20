import numbers
import warnings
from collections.abc import Iterable as IterableABC
from typing import Dict, Iterator, List, Type

from stats_arrays.distributions import (
    BernoulliUncertainty,
    BetaPERTUncertainty,
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


class UncertaintyChoices(IterableABC[Type[UncertaintyBase]]):
    """A container for uncertainty distributions, keyed by integer ID.

    Use :class:`stats_arrays.UncertaintyType` for named constants instead of
    raw integer IDs::

        uncertainty_choices[UncertaintyType.normal]   # NormalUncertainty
        uncertainty_choices[3]                        # same thing
    """

    #: Distributions in this container, sorted by ``id``.
    choices: List[Type[UncertaintyBase]]
    #: Lookup from integer ``id`` to distribution class.
    id_dict: Dict[int, Type[UncertaintyBase]]

    def __init__(self) -> None:
        # Sorted by id
        self.choices = sorted(DISTRIBUTIONS, key=lambda x: x.id)
        self.check_id_uniqueness()

    @staticmethod
    def check_id(distribution: Type[UncertaintyBase]) -> None:
        """Raise ``ValueError`` unless ``distribution.id`` is an integer that
        the ``uncertainty_type`` field of a params array can hold.

        Python and NumPy integers are accepted; booleans are not, although
        ``bool`` is a subclass of ``int``."""
        id_ = getattr(distribution, "id", None)
        if not isinstance(id_, numbers.Integral) or isinstance(id_, bool):
            raise ValueError(
                "Uncertainty distributions must have integer `id` attribute "
                f"(not a boolean). Got {id_!r}."
            )
        if not 0 <= int(id_) < 256:
            raise ValueError(
                "Uncertainty distribution ids must be between 0 and 255, as "
                "`uncertainty_type` in a params array is an unsigned 8-bit "
                f"integer. Got id {distribution.id}."
            )

    def check_id_uniqueness(self) -> None:
        """Rebuild ``id_dict`` from ``choices``, checking that each id is
        valid (see :meth:`check_id`) and used only once."""
        self.id_dict = {}
        for dist in self.choices:
            self.check_id(dist)
            id_ = int(dist.id)
            if id_ in self.id_dict:
                raise ValueError(
                    "Uncertainty id {} is already in use by {}".format(
                        id_, self.id_dict[id_].__name__
                    )
                )
            self.id_dict[id_] = dist

    def __iter__(self) -> Iterator[Type[UncertaintyBase]]:
        return iter(self.choices)

    def __getitem__(self, id_: int) -> Type[UncertaintyBase]:
        return self.id_dict[id_]

    def __len__(self) -> int:
        return len(self.id_dict)

    def __contains__(self, choice: Type[UncertaintyBase]) -> bool:
        return choice in self.choices

    def add(self, distribution: Type[UncertaintyBase]) -> None:
        self.check_id(distribution)
        id_ = int(distribution.id)
        if id_ in self.id_dict:
            warnings.warn(
                "ERROR: This distribution (id {}) is already present!".format(id_)
            )
            return
        # Keep ``choices`` sorted by id: ``MCRandomNumberGenerator`` relies on
        # it to match each distribution to its rows in the id-sorted params.
        self.choices.append(distribution)
        self.choices.sort(key=lambda x: x.id)
        self.id_dict[id_] = distribution


uncertainty_choices = UncertaintyChoices()
