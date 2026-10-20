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
