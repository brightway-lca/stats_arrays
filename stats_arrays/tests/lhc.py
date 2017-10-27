import unittest
from nose.plugins.skip import SkipTest
from numpy import *
from ..random import LatinHypercubeRNG
from ..distributions import *


class LatinHypercubeMCTestCase(unittest.TestCase):

    def make_params_array(self, length=1):
        assert isinstance(length, int)
        params = zeros((length,), dtype=[('uncertainty_type', 'i2'),
                                         ('input', 'u4'), ('output', 'u4'),
                                         ('loc', 'f4'), ('negative', 'b1'), ('scale', 'f4'),
                                         ('minimum', 'f4'), ('maximum', 'f4')])
        params['minimum'] = params['maximum'] = params['scale'] = NaN
        return params

    def test_known_seed(self):
        params = self.make_params_array(1)
        params['uncertainty_type'] = UniformUncertainty.id
        params['maximum'] = 1
        params['minimum'] = 0
        params['loc'] = 0.5
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, seed=11111, samples=9)
        self.assertTrue(allclose(next(lhc), array([[0.5]])))
        self.assertTrue(allclose(next(lhc), array([[0.7]])))
        self.assertTrue(allclose(next(lhc), array([[0.5]])))
        self.assertTrue(allclose(next(lhc), array([[0.6]])))

    def test_no_uncertainty_intervals(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = NoUncertainty.id
        params['loc'] = (1, 2)
        lhc = LatinHypercubeRNG(params)
        test_values = array(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
        self.assertTrue(allclose(lhc.hypercube, test_values))
        params['uncertainty_type'] = UndefinedUncertainty.id
        lhc = LatinHypercubeRNG(params)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['loc'] = 1
        params['uncertainty_type'] = NoUncertainty.id
        lhc = LatinHypercubeRNG(params)
        test_values = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))

    @SkipTest
    def test_lognormal_intervals(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = LognormalUncertainty.id
        params['loc'] = (1, 2)
        params['scale'] = (1, 2)
        lhc = LatinHypercubeRNG(params)
        test_values = array([[0.26311141, 0.40314545, 0.54630091, 0.70556548,
                              0.89209264, 1.12095981, 1.41730289, 1.83049303, 2.48049433, 3.8006714],
                             [0.52622281, 0.8062909, 1.09260181, 1.41113097, 1.78418529, 2.24191963,
                              2.83460579, 3.66098607, 4.96098868, 7.60134282]])
        # print lhc.hypercube
        # print test_values
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['loc'] = 1
        params['scale'] = 1
        params['uncertainty_type'] = LognormalUncertainty.id
        lhc = LatinHypercubeRNG(params)
        test_values = array([0.26311141, 0.40314545, 0.54630091, 0.70556548,
                             0.89209264, 1.12095981, 1.41730289, 1.83049303, 2.48049433, 3.8006714]
                            ).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Lower bound check
        params['loc'] = 1
        params['minimum'] = 1
        lhc = LatinHypercubeRNG(params)
        test_values = array([[1.12095981, 1.25845417, 1.41730289, 1.604463,
                              1.83049303, 2.11247151, 2.48049433, 2.99457872, 3.8006714, 5.42285066]
                             ]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Upper bound check
        params['loc'] = 1
        params['maximum'] = 1
        params['minimum'] = NaN
        lhc = LatinHypercubeRNG(params)
        test_values = array([[0.18440486, 0.26311141, 0.33393679, 0.40314545,
                              0.47337916, 0.54630091, 0.62326149, 0.70556548, 0.79462568,
                              0.89209264]]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Both bounds check
        params['loc'] = 1
        params['minimum'] = 0.5
        params['maximum'] = 1.5
        lhc = LatinHypercubeRNG(params)
        test_values = array([[0.56110473, 0.62514369, 0.69285366, 0.76502388,
                              0.84254518, 0.92646272, 1.01804071, 1.11884934, 1.23088804, 1.35676875]
                             ]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))

    @SkipTest
    def test_normal_intervals(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = NormalUncertainty.id
        params['loc'] = (1, 2)
        params['scale'] = (1, 2)
        lhc = LatinHypercubeRNG(params)
        test_values = array(([-0.33517774,  0.09154213,  0.39541465,  0.6512443,
                              0.88581471, 1.11418529, 1.3487557, 1.60458535, 1.90845787,
                              2.33517774], [-0.67035547,  0.18308426,  0.79082931,  1.30248861,
                                            1.77162941,  2.22837059,  2.69751139,  3.20917069,  3.81691574,
                                            4.67035547]))
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['loc'] = 1
        params['scale'] = 1
        params['uncertainty_type'] = NormalUncertainty.id
        lhc = LatinHypercubeRNG(params)
        test_values = array([-0.33517774,  0.09154213,  0.39541465,
                             0.6512443, 0.88581471,  1.11418529,  1.3487557,  1.60458535,
                             1.90845787, 2.33517774]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Lower bound check
        params['minimum'] = 1
        lhc = LatinHypercubeRNG(params)
        test_values = array([[1.11418529, 1.22988412, 1.3487557, 1.47278912,
                              1.60458535, 1.74785859, 1.90845787, 2.09680356, 2.33517774,
                              2.69062163]]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Upper bound check
        params['maximum'] = 1
        params['minimum'] = NaN
        lhc = LatinHypercubeRNG(params)
        test_values = array([[-0.69062163, -0.33517774, -0.09680356,
                              0.09154213, 0.25214141, 0.39541465, 0.52721088, 0.6512443,
                              0.77011588, 0.88581471]]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))
        # Both bounds check
        params['minimum'] = 0.5
        params['maximum'] = 1.5
        lhc = LatinHypercubeRNG(params)
        test_values = array([[0.59665956, 0.68968379, 0.78009253, 0.86873532,
                              0.95635658, 1.04364342, 1.13126468, 1.21990747, 1.31031621,
                              1.40334044]]).reshape(1, 10)
        self.assertTrue(allclose(lhc.hypercube, test_values))

    def test_uniform_intervals(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = UniformUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.5
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, samples=9)
        self.assertTrue(allclose(lhc.hypercube, array([[1.1, 1.2, 1.3, 1.4,
                                                        1.5, 1.6, 1.7, 1.8, 1.9], [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                                                                                   1.9]])))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['uncertainty_type'] = UniformUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.5
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, samples=9)
        self.assertTrue(allclose(lhc.hypercube,
                                 array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                         1.6, 1.7, 1.8, 1.9]])))

    def test_triangular_intervals(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = TriangularUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.6
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, samples=9)
        self.assertTrue(allclose(lhc.hypercube, array([[1.24494898, 1.34641017,
                                                        1.42426408, 1.48989796, 1.54772257, 1.60000001, 1.65358985, 1.7171573,
                                                        1.80000001], [1.24494898, 1.34641017, 1.42426408, 1.48989796,
                                                                      1.54772257, 1.60000001, 1.65358985, 1.7171573, 1.80000001]])))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['uncertainty_type'] = TriangularUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.6
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, samples=9)
        self.assertTrue(allclose(lhc.hypercube, array([[1.24494898, 1.34641017,
                                                        1.42426408, 1.48989796, 1.54772257, 1.60000001, 1.65358985, 1.7171573,
                                                        1.80000001]])))

    def test_bernoulli(self):
        # Two-dimensional array check
        params = self.make_params_array(2)
        params['uncertainty_type'] = BernoulliUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.6
        lhc = LatinHypercubeRNG(params)
        self.assertTrue(allclose(lhc.hypercube, array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                                                       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]])))
        # One-dimensional array check
        params = self.make_params_array(1)
        params['uncertainty_type'] = BernoulliUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.6
        lhc = LatinHypercubeRNG(params)
        self.assertTrue(allclose(lhc.hypercube, array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])))

    def test_heterogeneous_params(self):
        params = self.make_params_array(2)
        params['uncertainty_type'][0] = TriangularUncertainty.id
        params['uncertainty_type'][1] = UniformUncertainty.id
        params['maximum'] = 2
        params['minimum'] = 1
        params['loc'] = 1.6
        params['scale'] = NaN
        lhc = LatinHypercubeRNG(params, samples=9)
        self.assertTrue(allclose(lhc.hypercube, array([[1.24494898, 1.34641017,
                                                        1.42426408, 1.48989796, 1.54772257, 1.60000001, 1.65358985, 1.7171573,
                                                        1.80000001], [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])))
