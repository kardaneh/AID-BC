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

import numpy as np
import scipy.stats as sc

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.quantile_mapping import MonotoneInverse, QM, rv_histogram

# python -m unittest tests.test_quantile_mapping


# ============================================================================
# Unit Tests for MonotoneInverse
# ============================================================================


class TestMonotoneInverse(unittest.TestCase):
    """Unit tests for MonotoneInverse."""

    def setUp(self):
        """Create a test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_linear_inverse(self):
        """
        Test a monotone inverse using a linear function with a known inverse.

        For

        y = 2x + 3,

        the inverse is

        x = (y - 3) / 2.
        """
        self.logger.info("Testing MonotoneInverse with a linear function")

        def transform(x):
            return 2.0 * x + 3.0

        inverse = MonotoneInverse(
            xminmax=(0.0, 10.0),
            yminmax=(3.0, 23.0),
            transform=transform,
        )

        y = np.array([3.0, 7.0, 13.0, 23.0])

        expected = np.array(
            [
                (3.0 - 3.0) / 2.0,
                (7.0 - 3.0) / 2.0,
                (13.0 - 3.0) / 2.0,
                (23.0 - 3.0) / 2.0,
            ]
        )

        np.testing.assert_allclose(
            inverse(y),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

        self.logger.info("✅ MonotoneInverse linear-function test passed")


# ============================================================================
# Unit Tests for rv_histogram
# ============================================================================


class TestRvHistogram(unittest.TestCase):
    """Unit tests for the empirical distribution."""

    def setUp(self):
        """
        Fit an empirical distribution to ten ordered observations.

        The implementation computes:

        samples = [0, 10, 20, ..., 90]
        ranks   = [1, 2, 3, ..., 10]
        p       = [0, 0.2, 0.3, ..., 1]
        q       = [0, 10, 20, ..., 90]

        The first probability is replaced by zero by rv_histogram.
        """
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.samples = np.arange(
            0.0,
            100.0,
            10.0,
        )

        self.distribution = rv_histogram(X=self.samples)

    def test_cdf_quantiles(self):
        """Test empirical CDF values obtained from the sample ranks."""
        self.logger.info("Testing empirical CDF quantiles")

        values = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 90.0])

        # Interpolation points begin with:
        # (0, 0), (10, 0.2), (20, 0.3), (30, 0.4), ...
        expected_probabilities = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 1.0])

        np.testing.assert_allclose(
            self.distribution.cdf(values),
            expected_probabilities,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Empirical CDF quantiles test passed")

    def test_inverse_quantiles(self):
        """Test empirical inverse CDF values obtained by interpolation."""
        self.logger.info("Testing empirical inverse CDF quantiles")

        probabilities = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 1.0])

        expected_quantiles = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 90.0])

        np.testing.assert_allclose(
            self.distribution.ppf(probabilities),
            expected_quantiles,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Empirical inverse CDF quantiles test passed")


# ============================================================================
# Unit Tests for QM
# ============================================================================


class TestQuantileMapping(unittest.TestCase):
    """Unit tests for empirical and parametric quantile mapping."""

    def setUp(self):
        """Create a test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_empirical_mapping(self):
        """
        Test empirical quantile mapping.

        Biased sample:

        X = [0, 10, 20, ..., 90]

        Reference sample:

        Y = [100, 200, 300, ..., 1000]

        For example:

        - X = 5 has biased CDF probability 0.1.
        - The reference quantile at probability 0.1 is 150.
        - Therefore QM(5) = 150.
        """
        self.logger.info("Testing empirical Quantile Mapping")

        biased = np.arange(
            0.0,
            100.0,
            10.0,
        )

        reference = np.arange(
            100.0,
            1100.0,
            100.0,
        )

        values = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        expected = np.array([150.0, 200.0, 250.0, 300.0, 350.0])

        model = QM()
        model.fit(
            Y0=reference,
            X0=biased,
        )

        corrected = model.predict(values)

        self.assertEqual(
            corrected.shape,
            (values.size, 1),
        )

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Empirical Quantile Mapping test passed")

    def test_normal_mapping(self):
        """
        Test normal quantile mapping with frozen distributions.

        The source distribution is N(0, 1) and the reference distribution is
        N(10, 2). For a source value x, both distributions have the same
        standardized quantile, so the corrected value is:

        y = 10 + 2x.
        """
        self.logger.info("Testing normal-distribution Quantile Mapping")

        model = QM(
            n_features=1,
            distX0=sc.norm(
                loc=0.0,
                scale=1.0,
            ),
            distY0=sc.norm(
                loc=10.0,
                scale=2.0,
            ),
        )

        model.fit(
            Y0=None,
            X0=None,
        )

        values = np.array([-1.0, 0.0, 1.0])
        expected = np.array([8.0, 10.0, 12.0])

        corrected = model.predict(values)

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Normal-distribution Quantile Mapping test passed")

    def test_two_features(self):
        """
        Test independent correction of two features.

        Feature 1 maps N(0, 1) to N(10, 2):

        y1 = 10 + 2x1.

        Feature 2 maps N(100, 10) to N(-5, 5):

        y2 = -5 + 0.5 * (x2 - 100).
        """
        self.logger.info("Testing two-feature Quantile Mapping")

        model = QM(
            n_features=2,
            distX0=[
                sc.norm(
                    loc=0.0,
                    scale=1.0,
                ),
                sc.norm(
                    loc=100.0,
                    scale=10.0,
                ),
            ],
            distY0=[
                sc.norm(
                    loc=10.0,
                    scale=2.0,
                ),
                sc.norm(
                    loc=-5.0,
                    scale=5.0,
                ),
            ],
        )

        model.fit(
            Y0=None,
            X0=None,
        )

        values = np.array(
            [
                [-1.0, 90.0],
                [0.0, 100.0],
                [1.0, 110.0],
            ]
        )

        expected = np.array(
            [
                [8.0, -10.0],
                [10.0, -5.0],
                [12.0, 0.0],
            ]
        )

        corrected = model.predict(values)

        np.testing.assert_allclose(
            corrected,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Two-feature Quantile Mapping test passed")


def run_tests():
    """Run all quantile mapping tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMonotoneInverse))
    suite.addTests(loader.loadTestsFromTestCase(TestRvHistogram))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantileMapping))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
