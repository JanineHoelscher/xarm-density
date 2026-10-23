"""Numerical regression checks for the exact CDF-QP row-space reduction."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xarm_density.core import QuadraticCDFQP, ReducedQuadraticCDFQP


class ReducedQPTests(unittest.TestCase):
    def compare(self, dependent=False, cartesian=False, saturated=False):
        rng = np.random.default_rng(15)
        b = rng.normal(size=6)
        input_map = np.kron(np.eye(2), b[None, :])
        transport = (input_map[0] + input_map[1] if dependent
                     else rng.normal(size=12))
        nominal = np.array([2.0, -2.0]) if saturated else np.array([0.2, -0.1])
        task_map = np.array([[1.0, 0.2], [-0.1, 0.8]]) @ input_map if cartesian else input_map
        kwargs = dict(input_map=input_map, transport_row=transport,
                      nominal_input=nominal, task_map=task_map,
                      task_weight=1.0, input_weight=.02, theta_weight=.0002)
        results = [kind(12, 2, .01, .7).solve(**kwargs)
                   for kind in (QuadraticCDFQP, ReducedQuadraticCDFQP)]
        full, reduced = results
        np.testing.assert_allclose(full["input"], reduced["input"], atol=1e-5, rtol=1e-5)
        for result in results:
            self.assertGreaterEqual(result["residual"], .01 - 1e-7)
            self.assertLessEqual(np.abs(result["input"]).max(), .7 + 1e-7)
        def objective(result):
            theta = result["theta"]
            return (.5 * np.linalg.norm(task_map @ theta - nominal)**2
                    + .01 * np.linalg.norm(input_map @ theta)**2
                    + .0001 * np.linalg.norm(theta)**2)
        self.assertAlmostEqual(objective(full), objective(reduced), places=7)

    def test_joint_objective(self):
        self.compare()

    def test_cartesian_objective(self):
        self.compare(cartesian=True)

    def test_dependent_transport_row(self):
        self.compare(dependent=True)

    def test_active_command_bounds(self):
        self.compare(saturated=True)

    def test_unsupported_regularization(self):
        solver = ReducedQuadraticCDFQP(12, 2, .01, .7)
        with self.assertRaises(ValueError):
            solver.solve(np.ones((2, 12)), np.ones(12), np.zeros(2), theta_weight=0)

    def test_task_outside_row_space(self):
        rng = np.random.default_rng(6)
        solver = ReducedQuadraticCDFQP(12, 2, .01, .7)
        with self.assertRaises(ValueError):
            solver.solve(rng.normal(size=(2, 12)), rng.normal(size=12), np.zeros(2),
                         task_map=rng.normal(size=(2, 12)))


if __name__ == "__main__":
    unittest.main()
