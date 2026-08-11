# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import unittest

# Force CPU execution for lightweight and reproducible unit tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.optimal_transport import SinkhornOutput, SinkhornSolver


# python -m unittest tests.test_optimal_transport


# ============================================================================
# Unit Tests for SinkhornOutput
# ============================================================================


class TestSinkhornOutput(unittest.TestCase):
    """Unit tests for SinkhornOutput."""

    def setUp(self):
        """Create a test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_transport_plan(self):
        """Test transport-plan construction and epsilon validation."""
        self.logger.info("Testing SinkhornOutput transport plan")

        output = SinkhornOutput(
            potentials=(
                jnp.zeros(2),
                jnp.zeros(2),
            ),
            cost_matrix=jnp.zeros((2, 2)),
            epsilon=1.0,
        )

        expected = np.ones((2, 2))

        np.testing.assert_allclose(
            np.asarray(output.transport_plan),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        invalid_output = SinkhornOutput(
            potentials=(
                jnp.zeros(2),
                jnp.zeros(2),
            ),
            cost_matrix=jnp.zeros((2, 2)),
            epsilon=0.0,
        )

        with self.assertRaises(ValueError):
            _ = invalid_output.transport_plan

        self.logger.info("✅ SinkhornOutput transport-plan test passed")


# ============================================================================
# Unit Tests for SinkhornSolver
# ============================================================================


class TestSinkhornSolver(unittest.TestCase):
    """Unit tests for the Sinkhorn optimal transport solver."""

    def setUp(self):
        """Create a test logger and a small Sinkhorn solver."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.solver = SinkhornSolver(
            epsilon=1.0,
            num_iterations=100,
            threshold=1e-6,
        )

    def test_compute_cost(self):
        """Test the pairwise squared Euclidean distance matrix."""
        self.logger.info("Testing Sinkhorn squared Euclidean cost matrix")

        x = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
            ]
        )

        y = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
            ]
        )

        expected = np.array(
            [
                [0.0, 4.0],
                [5.0, 5.0],
            ]
        )

        cost = self.solver._compute_cost(x, y)

        np.testing.assert_allclose(
            np.asarray(cost),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Sinkhorn cost-matrix test passed")

    def test_invalid_dimensions(self):
        """Test rejection of non-two-dimensional input arrays."""
        self.logger.info("Testing Sinkhorn input-dimension validation")

        x = jnp.array([0.0, 1.0])
        y = jnp.array([[0.0], [1.0]])

        with self.assertRaises(ValueError):
            self.solver._forward_solve(x, y)

        self.logger.info("✅ Sinkhorn input-dimension validation test passed")

    def test_small_sinkhorn_problem(self):
        """Test the complete Sinkhorn solver on a very small problem."""
        self.logger.info("Testing Sinkhorn solver on a small problem")

        x = jnp.array(
            [
                [0.0],
                [1.0],
            ]
        )

        y = jnp.array(
            [
                [0.0],
                [1.0],
            ]
        )

        output = self.solver(x, y)

        self.assertEqual(
            output.cost_matrix.shape,
            (2, 2),
        )

        self.assertEqual(
            output.fu.shape,
            (2,),
        )

        self.assertEqual(
            output.gv.shape,
            (2,),
        )

        self.assertLessEqual(
            int(output.num_iterations),
            self.solver.num_iterations,
        )

        self.assertTrue(np.isfinite(np.asarray(output.reg_ot_cost)).all())

        np.testing.assert_allclose(
            np.asarray(output.cost_matrix),
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Small Sinkhorn solver test passed")


def run_tests():
    """Run all optimal transport tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSinkhornOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestSinkhornSolver))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
