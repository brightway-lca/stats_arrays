import unittest
from stats_arrays.uncertainty_choices import *


class UncertaintyChoicesTestCase(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(UndefinedUncertainty in uncertainty_choices)
