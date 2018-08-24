import unittest
from numpy import *
from ..random import RandomNumberGenerator as RNG
from ..distributions import *
from ..errors import UnknownUncertaintyType, ImproperBoundsError
import numpy as np


class RandomNumberGeneratorTestCase(unittest.TestCase):

    def test_invalid_uncertainty_type(self):
        with self.assertRaises(UnknownUncertaintyType):
            RNG(object, UncertaintyBase.from_dicts({}))

    def test_uncertainty_not_subclass(self):
        class Foo(object):

            @classmethod
            def bounded_random_variables(self):
                pass

            @classmethod
            def validate(self, *args, **kwargs):
                pass

        RNG(Foo, UncertaintyBase.from_dicts({}))

    def test_method_call(self):
        rng = RNG(
            NormalUncertainty,
            UncertaintyBase.from_dicts({'loc': 0, 'scale': 1})
        )
        rng.next()
        rng.generate_random_numbers()

    def test_as_iterator(self):
        counter = 0
        rng = RNG(
            NormalUncertainty,
            UncertaintyBase.from_dicts({'loc': 0, 'scale': 1})
        )
        for x in rng:
            counter += 1
            if counter >= 10:
                break

    def test_seed(self):
        data = []
        for x in range(2):
            rng = RNG(
                NormalUncertainty,
                UncertaintyBase.from_dicts({'loc': 0, 'scale': 1}),
                seed=111
            )
            data.append(rng.next())
        self.assertTrue(np.allclose(*data))

    def test_output_dimensions(self):
        rng = RNG(
            NormalUncertainty,
            UncertaintyBase.from_dicts(
                {'loc': 0, 'scale': 1},
                {'loc': 1, 'scale': 2}
            ),
            size=10
        )
        self.assertTrue(rng.next().shape, (2, 10))

    def test_validation(self):
        with self.assertRaises(ImproperBoundsError):
            rng = RNG(
                TriangularUncertainty,
                UncertaintyBase.from_dicts({'loc': -0.000000001, 'minimum': 0, 'maximum': 1})
            )
        # No error
        rng = RNG(
            TriangularUncertainty,
            UncertaintyBase.from_dicts({'loc': 0.5, 'minimum': 0, 'maximum': 1})
        )
        rng = RNG(
            TriangularUncertainty,
            UncertaintyBase.from_dicts({'loc': 0., 'minimum': 0, 'maximum': 1})
        )
        rng = RNG(
            TriangularUncertainty,
            UncertaintyBase.from_dicts({'loc': 1., 'minimum': 0, 'maximum': 1})
        )
