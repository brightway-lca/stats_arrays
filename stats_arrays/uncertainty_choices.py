from .distributions import *
import warnings

DISTRIBUTIONS = (
    UndefinedUncertainty,
    NoUncertainty,
    LognormalUncertainty,
    NormalUncertainty,
    UniformUncertainty,
    TriangularUncertainty,
    BernoulliUncertainty,
    BetaUncertainty,
    DiscreteUniform,
    GammaUncertainty,
    WeibullUncertainty,
)


class UncertaintyChoices(object):
    """An container for uncertainty distributions."""
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

    def add(self, distribution):
        if not hasattr(distribution, "id") and isinstance(distributions.id, int):
            raise ValueError("Uncertainty distributions must have integer `id` attribute.")
        if distribution.id in self.id_dict:
            warnings.warn("This distribution (id %s) is already present" % distribution.id)
            return
        self.choices.append(distribution)
        self.id_dict[distribution.id] = distribution

uncertainty_choices = UncertaintyChoices()
