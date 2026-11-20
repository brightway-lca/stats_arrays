import numpy as np
import pytest

from stats_arrays.distributions import NormalUncertainty, UncertaintyBase
from stats_arrays.uncertainty_choices import (
    BetaPERTUncertainty,
    UncertaintyChoices,
    UndefinedUncertainty,
    uncertainty_choices,
)


def test_contains():
    assert UndefinedUncertainty in uncertainty_choices
    assert BetaPERTUncertainty in uncertainty_choices


def test_add_new_distribution():
    class Custom(UncertaintyBase):
        id = 1000
        description = "Custom"

    choices = UncertaintyChoices()
    choices.add(Custom)
    assert Custom in choices
    assert choices[1000] is Custom


def test_add_rejects_missing_id():
    class NoId(UncertaintyBase):
        description = "No id"

    with pytest.raises(ValueError, match="integer `id`"):
        UncertaintyChoices().add(NoId)


@pytest.mark.parametrize("bad_id", [None, "1001", 1001.0])
def test_add_rejects_non_integer_id(bad_id):
    class BadId(UncertaintyBase):
        id = bad_id
        description = "Bad id"

    choices = UncertaintyChoices()
    with pytest.raises(ValueError, match="integer `id`"):
        choices.add(BadId)
    assert BadId not in choices


def test_add_existing_id_warns():
    choices = UncertaintyChoices()
    with pytest.warns(UserWarning, match=r"id {}\)".format(NormalUncertainty.id)):
        choices.add(NormalUncertainty)
    assert choices.choices.count(NormalUncertainty) == 1


def test_duplicate_id_error_message():
    class Duplicate(UncertaintyBase):
        id = NormalUncertainty.id
        description = "Duplicate"

    choices = UncertaintyChoices()
    choices.choices.append(Duplicate)
    with pytest.raises(
        ValueError,
        match="id {} is already in use by NormalUncertainty".format(
            NormalUncertainty.id
        ),
    ):
        choices.check_id_uniqueness()


def test_add_keeps_choices_sorted_by_id():
    class Low(UncertaintyBase):
        id = -5
        description = "Low"

    class Middle(UncertaintyBase):
        id = 50
        description = "Middle"

    class High(UncertaintyBase):
        id = 99
        description = "High"

    choices = UncertaintyChoices()
    for distribution in (High, Middle, Low):
        choices.add(distribution)
    ids = [distribution.id for distribution in choices]
    assert ids == sorted(ids)
    assert ids[0] == -5 and ids[-2:] == [50, 99]


def test_mc_rows_match_distributions_after_add(monkeypatch):
    """Regression test for #49: adding a distribution with a lower id than an
    existing one made ``MCRandomNumberGenerator`` sample the wrong rows."""
    from stats_arrays import MCRandomNumberGenerator

    class Middle(UndefinedUncertainty):
        id = 99

    class Low(NormalUncertainty):
        id = 50

    choices = UncertaintyChoices()
    choices.add(Middle)
    choices.add(Low)
    monkeypatch.setattr("stats_arrays.random.uncertainty_choices", choices)

    params = UncertaintyBase.from_dicts(
        {"loc": 7, "uncertainty_type": 99},
        {"loc": 7, "uncertainty_type": 99},
        {"loc": 1000, "scale": 1, "uncertainty_type": 50},
        {"loc": 3, "scale": 0.1, "uncertainty_type": NormalUncertainty.id},
    )
    data = MCRandomNumberGenerator(params, seed=1).generate(samples=50)
    assert (data[0] == 7).all() and (data[1] == 7).all()
    assert np.abs(data[2] - 1000).max() < 10
    assert np.abs(data[3] - 3).max() < 1
