from stats_arrays.uncertainty_choices import UndefinedUncertainty, uncertainty_choices, BetaPERTUncertainty


def test_contains():
    assert UndefinedUncertainty in uncertainty_choices
    assert BetaPERTUncertainty in uncertainty_choices

