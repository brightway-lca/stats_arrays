from stats_arrays.uncertainty_choices import UndefinedUncertainty, uncertainty_choices


def test_contains():
    assert UndefinedUncertainty in uncertainty_choices
