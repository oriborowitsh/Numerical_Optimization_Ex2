import unittest
import numpy as np
from src.utils import *
from src.constrained_min import interior_pt
from tests.examples import *
from src.utils import *

class TestUnconstrainedMin(unittest.TestCase):
    def test_qp(self):
        ineq_constraints=[quadratic_example_g1,quadratic_example_g2,quadratic_example_g3]
        x_opt, f_opt, history=interior_pt(quadratic_example_f, ineq_constraints,
                    quadratic_example_lhs, quadratic_example_rhs, np.array([0.1, 0.2, 0.7]))

        triangle = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        limits=([0, 1], [0, 1], [0, 1])

        plot_central_path(triangle, history, limits, "Quadratic Programming")
        plot_function_iterations(history, "Quadratic Programming")

    def test_lp(self):
        ineq_constraints = [linear_example_g1, linear_example_g2, linear_example_g3, linear_example_g4]
        x_opt, f_opt, history = interior_pt(linear_example_f, ineq_constraints, None, None,
                            np.array([0.5, 0.75]))

        polygon = [
            [0.0, 1.0],
            [2.0, 1.0],
            [2.0, 0.0],
            [1.0, 0.0]
        ]
        limits=([0, 2.2], [0, 1.2])

        plot_central_path(polygon, history, limits, "Linear Programming")
        plot_function_iterations(history, "Linear Programming")

if __name__ == '__main__':
    unittest.main()