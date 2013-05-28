from .distributions import *

DEFAULT_DISTRIBUTIONS = (
    UndefinedUncertainty,
    NoUncertainty,
    LognormalUncertainty,
    NormalUncertainty,
    UniformUncertainty,
    TriangularUncertainty,
    BernoulliUncertainty,
    BetaUncertainty,
    DiscreteUniform
)
CUSTOM_DISTRIBUTIONS = ()
DISTRIBUTIONS = DEFAULT_DISTRIBUTIONS + CUSTOM_DISTRIBUTIONS


class UncertaintyChoices(object):
    """An iterable for uncertainty choices"""
    def __init__(self):
        # Sorted by id
        self.choices = sorted(DISTRIBUTIONS, key=lambda x: x.id)
        self.check_id_uniqueness()

    def check_id_uniqueness(self):
        self.id_dict = {}
        for dist in self.choices:
            if dist.id in self.id_dict:
                raise ValueError("Uncertainty id %i is already in use by %s" %
                                (dist.id, self.id_dict[dist.id]))
            self.id_dict[dist.id] = dist

    def __iter__(self):
        return iter(self.choices)

    def __getitem__(self, index):
        return self.id_dict[index]

    def __len__(self):
        return len(self.id_dict)

    @property
    def choices_tuple(self):
        """Formatted for Django ChoiceField"""
        return [(obj.id, obj.description) for obj in self.choices]

uncertainty_choices = UncertaintyChoices()
