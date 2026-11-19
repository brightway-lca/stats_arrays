"""Tests for ``numpy.random.Generator`` support and the ``generate`` fast paths (#46)."""

import numpy as np
import pytest

from stats_arrays import (
    BernoulliUncertainty,
    BetaPERTUncertainty,
    BetaUncertainty,
    DiscreteUniform,
    GammaUncertainty,
    GeneralizedExtremeValueUncertainty,
    LatinHypercubeRNG,
    LognormalUncertainty,
    MCRandomNumberGenerator,
    NormalUncertainty,
    NoUncertainty,
    RandomNumberGenerator,
    StudentsTUncertainty,
    TriangularUncertainty,
    UncertaintyBase,
    UniformUncertainty,
    WeibullUncertainty,
)
from stats_arrays.utils import make_random_source

ROWS = [
    {"uncertainty_type": NoUncertainty.id, "loc": 1},
    {"uncertainty_type": LognormalUncertainty.id, "loc": 0.5, "scale": 0.2},
    {
        "uncertainty_type": LognormalUncertainty.id,
        "loc": 0.5,
        "scale": 0.2,
        "negative": True,
    },
    {
        "uncertainty_type": LognormalUncertainty.id,
        "loc": 0,
        "scale": 1,
        "minimum": 0.5,
        "maximum": 2,
    },
    {"uncertainty_type": NormalUncertainty.id, "loc": 10, "scale": 0.5},
    {"uncertainty_type": UniformUncertainty.id, "minimum": 1, "maximum": 2},
    {
        "uncertainty_type": TriangularUncertainty.id,
        "loc": 1.5,
        "minimum": 1,
        "maximum": 2,
    },
    {"uncertainty_type": BernoulliUncertainty.id, "loc": 0.5},
    {"uncertainty_type": BernoulliUncertainty.id, "loc": 0.0},
    {"uncertainty_type": DiscreteUniform.id, "minimum": 3, "maximum": 6},
    {"uncertainty_type": WeibullUncertainty.id, "loc": 0, "scale": 1, "shape": 2},
    {"uncertainty_type": GammaUncertainty.id, "loc": 0, "scale": 1, "shape": 2},
    {
        "uncertainty_type": BetaUncertainty.id,
        "loc": 2,
        "shape": 3,
        "minimum": 0,
        "maximum": 1,
    },
    {
        "uncertainty_type": GeneralizedExtremeValueUncertainty.id,
        "loc": 0,
        "scale": 1,
        "shape": 0,
    },
    {"uncertainty_type": StudentsTUncertainty.id, "loc": 0, "scale": 1, "shape": 3},
    {"uncertainty_type": StudentsTUncertainty.id, "loc": 100, "scale": 1, "shape": 3},
    {
        "uncertainty_type": BetaPERTUncertainty.id,
        "loc": 1,
        "minimum": 0,
        "maximum": 3,
    },
]


@pytest.fixture()
def params():
    return UncertaintyBase.from_dicts(*ROWS)


# make_random_source


def test_make_random_source_default_is_generator():
    assert isinstance(make_random_source(), np.random.Generator)


@pytest.mark.parametrize(
    "seed",
    [1, np.int64(1), [1, 2], np.random.SeedSequence(1), np.random.PCG64(1)],
)
def test_make_random_source_builds_generator(seed):
    assert isinstance(make_random_source(seed), np.random.Generator)


def test_make_random_source_integer_seed_is_default_rng():
    assert make_random_source(1).random() == np.random.default_rng(1).random()


def test_make_random_source_passes_instances_through():
    g = np.random.default_rng(1)
    rs = np.random.RandomState(1)
    assert make_random_source(g) is g
    assert make_random_source(rs) is rs


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_make_random_source_rejects_bad_seed(seed):
    with pytest.raises(TypeError):
        make_random_source(seed)


# seeded_random type check


def test_bounded_random_variables_accepts_generator():
    p = NormalUncertainty.from_dicts({"loc": 0, "scale": 1, "minimum": -1})
    data = NormalUncertainty.bounded_random_variables(p, 100, np.random.default_rng(1))
    assert data.shape == (1, 100)
    assert (data >= -1).all()


def test_bounded_random_variables_rejects_int():
    p = NormalUncertainty.from_dicts({"loc": 0, "scale": 1})
    with pytest.raises(TypeError):
        NormalUncertainty.bounded_random_variables(p, 1, 42)


# MCRandomNumberGenerator

# Factories, so each generator gets a fresh, identically seeded source
SEEDS = [
    pytest.param(lambda: 3, id="int"),
    pytest.param(lambda: np.random.default_rng(3), id="Generator"),
    pytest.param(lambda: np.random.RandomState(3), id="RandomState"),
]


@pytest.mark.parametrize("seed", SEEDS)
def test_mc_reproducible(params, seed):
    a = MCRandomNumberGenerator(params, seed=seed()).generate(samples=5)
    b = MCRandomNumberGenerator(params, seed=seed()).generate(samples=5)
    assert a.shape == (len(ROWS), 5)
    assert np.array_equal(a, b)


def test_mc_integer_seed_uses_generator(params):
    assert isinstance(
        MCRandomNumberGenerator(params, seed=1).random, np.random.Generator
    )
    assert isinstance(MCRandomNumberGenerator(params).random, np.random.Generator)


def test_mc_random_state_seed_gives_legacy_stream():
    p = NormalUncertainty.from_dicts(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"loc": 5, "scale": 2, "uncertainty_type": NormalUncertainty.id},
    )
    data = MCRandomNumberGenerator(p, seed=np.random.RandomState(1)).generate()
    expected = np.random.RandomState(1).normal(p["loc"], p["scale"], size=(1, 2))[0]
    assert np.array_equal(data, expected)


def test_mc_generator_differs_from_random_state(params):
    a = MCRandomNumberGenerator(params, seed=np.random.RandomState(1)).generate()
    b = MCRandomNumberGenerator(params, seed=1).generate()
    assert not np.array_equal(a, b)


def test_mc_generator_values_respect_params(params):
    data = MCRandomNumberGenerator(params, seed=7).generate(samples=2000)
    assert (data[0] == 1).all()
    assert (data[1] > 0).all()
    assert (data[2] < 0).all()
    assert ((data[3] >= 0.5) & (data[3] <= 2)).all()
    assert abs(np.median(data[4]) - 10) < 0.1
    assert ((data[5] >= 1) & (data[5] <= 2)).all()
    assert ((data[6] >= 1) & (data[6] <= 2)).all()
    assert set(np.unique(data[7])) == {0, 1}
    assert (data[8] == 0).all()
    assert set(np.unique(data[9])) == {3, 4, 5}
    assert (data[10] > 0).all() and (data[11] > 0).all()
    assert ((data[12] >= 0) & (data[12] <= 1)).all()
    assert abs(np.median(data[15]) - 100) < 1
    assert ((data[16] >= 0) & (data[16] <= 3)).all()


def test_mc_spawned_seed_sequences_are_independent(params):
    children = np.random.SeedSequence(1).spawn(2)
    a = MCRandomNumberGenerator(params, seed=children[0]).generate()
    b = MCRandomNumberGenerator(params, seed=children[1]).generate()
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("samples", [1, 4])
def test_generate_out(params, seed, samples):
    expected = MCRandomNumberGenerator(params, seed=seed()).generate(samples)
    mc = MCRandomNumberGenerator(params, seed=seed())
    out = np.full(expected.shape, np.nan)
    result = mc.generate(samples, out=out)
    assert result is out
    assert np.array_equal(out, expected)


def test_generate_out_wrong_shape(params):
    mc = MCRandomNumberGenerator(params)
    with pytest.raises(ValueError):
        mc.generate(out=np.empty(len(ROWS) + 1))
    with pytest.raises(ValueError):
        mc.generate(samples=3, out=np.empty(len(ROWS)))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("samples", [1, 4])
def test_generate_sorted_order(params, seed, samples):
    original = MCRandomNumberGenerator(params, seed=seed()).generate(samples)
    mc = MCRandomNumberGenerator(params, seed=seed())
    in_sorted = mc.generate(samples, order="sorted")
    assert np.array_equal(in_sorted, original[mc.ordering])
    assert np.array_equal(in_sorted[mc.reverse_ordering], original)


def test_generate_bad_order(params):
    with pytest.raises(ValueError):
        MCRandomNumberGenerator(params).generate(order="random")


def test_generate_single_type_identity_ordering():
    p = LognormalUncertainty.from_dicts(
        *[{"loc": i, "scale": 0.01, "uncertainty_type": 2} for i in range(5)]
    )
    mc = MCRandomNumberGenerator(p, seed=1)
    data = mc.generate()
    assert np.allclose(np.log(data), np.arange(5), atol=0.1)


def test_bounds_set_after_instantiation_are_respected():
    p = UncertaintyBase.from_dicts(
        {"loc": 0, "scale": 1, "uncertainty_type": NormalUncertainty.id},
        {"loc": 0, "scale": 1, "uncertainty_type": LognormalUncertainty.id},
    )
    mc = MCRandomNumberGenerator(p, seed=1)
    mc.generate()
    mc.params["minimum"][:] = 0.5
    data = mc.generate(samples=200)
    assert (data >= 0.5).all()


@pytest.mark.parametrize("decorator", [classmethod, staticmethod])
def test_custom_bounded_random_variables_is_called(params, monkeypatch, decorator):
    """Overrides are used whether they are class methods or static methods, and
    whether they are set before or after instantiation."""
    calls = []

    def custom(*args, **kwargs):
        calls.append(1)
        params = args[-4]
        return np.zeros((params.shape[0], args[-3]))

    before = MCRandomNumberGenerator(params)
    monkeypatch.setattr(
        NormalUncertainty, "bounded_random_variables", decorator(custom)
    )
    after = MCRandomNumberGenerator(params)
    assert before.generate()[4] == 0
    assert after.generate()[4] == 0
    assert len(calls) == 2


# Other generators


def test_random_number_generator_integer_seed_uses_generator():
    p = LognormalUncertainty.from_dicts({"loc": 0, "scale": 1})
    a = RandomNumberGenerator(LognormalUncertainty, p, size=3, seed=5)
    b = RandomNumberGenerator(LognormalUncertainty, p, size=3, seed=5)
    assert isinstance(a.random, np.random.Generator)
    assert np.array_equal(a.generate_random_numbers(), b.generate_random_numbers())


def test_latin_hypercube_integer_seed_uses_generator():
    p = UncertaintyBase.from_dicts(
        {"loc": 1, "scale": 0.5, "uncertainty_type": NormalUncertainty.id},
        {"minimum": 0, "maximum": 1, "uncertainty_type": UniformUncertainty.id},
    )
    a = LatinHypercubeRNG(p, seed=5, samples=10)
    b = LatinHypercubeRNG(p, seed=5, samples=10)
    assert isinstance(a.random, np.random.Generator)
    first = a.next()
    assert np.array_equal(first, b.next())
    for row in range(2):
        assert first[row] in a.hypercube[row]


# Bug fixes found while adding Generator support


@pytest.mark.parametrize("seed", SEEDS)
def test_bernoulli_multiple_rows(seed):
    p = BernoulliUncertainty.from_dicts({"loc": 0}, {"loc": 1}, {"loc": 0.5})
    rs = make_random_source(seed())
    data = BernoulliUncertainty.random_variables(p, 1000, rs)
    assert data.shape == (3, 1000)
    assert (data[0] == 0).all()
    assert (data[1] == 1).all()
    assert 0.4 < data[2].mean() < 0.6


@pytest.mark.parametrize("samples", [1, 5])
def test_students_t_per_row_loc_and_scale(samples):
    p = StudentsTUncertainty.from_dicts(
        {"loc": 0, "scale": 1, "shape": 30},
        {"loc": 1000, "scale": 0.001, "shape": 30},
    )
    data = StudentsTUncertainty.random_variables(p, samples, np.random.RandomState(1))
    assert data.shape == (2, samples)
    assert (np.abs(data[1] - 1000) < 1).all()
    assert (np.abs(data[0]) < 100).all()
