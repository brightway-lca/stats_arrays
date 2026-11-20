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
        id = 200
        description = "Custom"

    choices = UncertaintyChoices()
    choices.add(Custom)
    assert Custom in choices
    assert choices[200] is Custom


@pytest.mark.parametrize("bad_id", [-5, -1, 256, 1000])
def test_add_rejects_id_outside_uint8(bad_id):
    """#64: ``uncertainty_type`` is an unsigned 8-bit field."""

    class OutOfRange(UncertaintyBase):
        id = bad_id
        description = "Out of range"

    choices = UncertaintyChoices()
    with pytest.raises(ValueError, match=rf"between 0 and 255.*Got id {bad_id}\."):
        choices.add(OutOfRange)
    assert OutOfRange not in choices
    assert bad_id not in choices.id_dict


@pytest.mark.parametrize("good_id", [0, 255])
def test_add_accepts_uint8_limits(good_id):
    class Limit(UncertaintyBase):
        id = good_id
        description = "Limit"

    choices = UncertaintyChoices()
    choices.id_dict.pop(good_id, None)
    choices.choices = [d for d in choices.choices if d.id != good_id]
    choices.add(Limit)
    assert choices[good_id] is Limit


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


def test_check_id_uniqueness_rejects_id_outside_uint8():
    """The range rule holds for distributions that didn't come through ``add()``."""

    class OutOfRange(UncertaintyBase):
        id = 1000
        description = "Out of range"

    choices = UncertaintyChoices()
    choices.choices.append(OutOfRange)
    with pytest.raises(ValueError, match=r"between 0 and 255.*Got id 1000\."):
        choices.check_id_uniqueness()
    assert 1000 not in choices.id_dict


def test_add_keeps_choices_sorted_by_id():
    class Low(UncertaintyBase):
        id = 20
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
    assert ids[-3:] == [20, 50, 99]


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
