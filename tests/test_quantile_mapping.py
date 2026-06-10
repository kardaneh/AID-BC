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

from AID_BC.quantile_mapping import MonotoneInverse, QM, rv_histogram

# python -m unittest tests.test_quantile_mapping


# ============================================================================
# Unit Tests for MonotoneInverse
# ============================================================================


class TestMonotoneInverse(unittest.TestCase):
    """Unit tests for MonotoneInverse."""

    def test_monotone_inverse_with_linear_function(self):
        """Test inversion of a strictly increasing linear function."""

        # Define a monotonic linear transformation with a known inverse:
        # y = 2x + 3  =>  x = (y - 3) / 2.
        def transform(x):
            """Apply the linear transformation y = 2x + 3."""

            return 2.0 * x + 3.0

        inverse = MonotoneInverse(
            xminmax=(0.0, 10.0),
            yminmax=(3.0, 23.0),
            transform=transform,
        )

        # Evaluate the numerical inverse at values with exact expected results.
        y = np.array([3.0, 7.0, 13.0, 23.0])
        expected = np.array([0.0, 2.0, 5.0, 10.0])

        result = inverse(y)

        # A tight tolerance is appropriate because the transformation is linear.
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_monotone_inverse_extends_initial_bounds(self):
        """Test automatic extension of the initial x bounds."""

        # The identity function makes it easy to verify that the constructor
        # extends the initial x range to cover the requested y range.
        def transform(x):
            """Return the input unchanged."""

            return x

        inverse = MonotoneInverse(
            xminmax=(0.0, 1.0),
            yminmax=(-0.5, 1.5),
            transform=transform,
        )

        # The original [0, 1] interval must be expanded to cover [-0.5, 1.5].
        self.assertLessEqual(inverse.xmin, -0.5)
        self.assertGreaterEqual(inverse.xmax, 1.5)

        result = inverse(np.array([-0.5, 0.0, 1.0, 1.5]))

        np.testing.assert_allclose(
            result,
            np.array([-0.5, 0.0, 1.0, 1.5]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_monotone_inverse_accepts_scalar(self):
        """Test inversion of a scalar value."""

        # For y = 4x - 2, the value y = 2 corresponds to x = 1.
        def transform(x):
            """Apply the linear transformation y = 4x - 2."""

            return 4.0 * x - 2.0

        inverse = MonotoneInverse(
            xminmax=(-2.0, 2.0),
            yminmax=(-10.0, 6.0),
            transform=transform,
        )

        result = inverse(2.0)

        self.assertAlmostEqual(float(result), 1.0, places=10)


# ============================================================================
# Unit Tests for rv_histogram
# ============================================================================


class TestRvHistogram(unittest.TestCase):
    """Unit tests for the empirical rv_histogram distribution."""

    def setUp(self):
        """Create deterministic samples for empirical distribution tests."""

        # Use evenly spaced deterministic data to make the tests reproducible.
        self.samples = np.linspace(-5.0, 5.0, 201)

        # Build the empirical CDF, inverse CDF and PDF from the sample.
        self.distribution = rv_histogram(X=self.samples)

    def test_initialization_from_samples(self):
        """Test empirical distribution initialization from samples."""

        # All interpolation functions must be created during initialization.
        self.assertIsNotNone(self.distribution._cdf)
        self.assertIsNotNone(self.distribution._icdf)
        self.assertIsNotNone(self.distribution._pdf)

    def test_cdf_is_monotonically_increasing(self):
        """Test that the empirical CDF is monotonically increasing."""

        x = np.linspace(-6.0, 6.0, 500)
        probabilities = self.distribution.cdf(x)

        # A valid cumulative distribution function must never decrease.
        self.assertTrue(np.all(np.diff(probabilities) >= 0.0))

    def test_cdf_bounds(self):
        """Test CDF values outside the fitted sample range."""

        # Values below and above the observed support should be mapped
        # to the limiting probabilities 0 and 1.
        self.assertEqual(float(self.distribution.cdf(-100.0)), 0.0)
        self.assertEqual(float(self.distribution.cdf(100.0)), 1.0)

    def test_icdf_bounds(self):
        """Test inverse CDF at probability boundaries."""

        # The inverse CDF at probability 0 and 1 should return the
        # minimum and maximum observed values.
        self.assertAlmostEqual(
            float(self.distribution.icdf(0.0)),
            self.samples.min(),
        )

        self.assertAlmostEqual(
            float(self.distribution.icdf(1.0)),
            self.samples.max(),
        )

    def test_cdf_and_icdf_are_consistent(self):
        """Test approximate consistency between CDF and inverse CDF."""

        probabilities = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        # Convert probabilities to quantiles, then map them back through the CDF.
        quantiles = self.distribution.icdf(probabilities)
        recovered_probabilities = self.distribution.cdf(quantiles)

        # The empirical resolution is approximately one sample probability step.
        np.testing.assert_allclose(
            recovered_probabilities,
            probabilities,
            atol=1.0 / self.samples.size,
        )

    def test_ppf_is_alias_of_icdf(self):
        """Test that ppf returns the same result as icdf."""

        probabilities = np.array([0.1, 0.5, 0.9])

        # In scipy terminology, ppf is another name for the inverse CDF.
        np.testing.assert_allclose(
            self.distribution.ppf(probabilities),
            self.distribution.icdf(probabilities),
        )

    def test_survival_function(self):
        """Test that sf is equal to one minus the CDF."""

        x = np.array([-2.0, 0.0, 2.0])

        np.testing.assert_allclose(
            self.distribution.sf(x),
            1.0 - self.distribution.cdf(x),
        )

    def test_inverse_survival_function(self):
        """Test that isf uses the complementary probability."""

        probabilities = np.array([0.1, 0.25, 0.75])

        # By definition, isf(p) is equivalent to icdf(1 - p).
        np.testing.assert_allclose(
            self.distribution.isf(probabilities),
            self.distribution.icdf(1.0 - probabilities),
        )

    def test_pdf_is_zero_outside_sample_range(self):
        """Test that the empirical PDF is zero outside its support."""

        values = self.distribution.pdf(np.array([-100.0, 100.0]))

        # The configured interpolation fill value is zero outside the histogram.
        np.testing.assert_allclose(
            values,
            np.array([0.0, 0.0]),
        )

    def test_random_variates_shape_and_bounds(self):
        """Test generated random variates have the expected shape and range."""

        # Fix the random seed so the test remains reproducible.
        np.random.seed(42)

        samples = self.distribution.rvs(size=100)

        self.assertEqual(samples.shape, (100,))

        # Generated values must remain inside the empirical distribution support.
        self.assertTrue(np.all(samples >= self.samples.min()))
        self.assertTrue(np.all(samples <= self.samples.max()))


# ============================================================================
# Unit Tests for QM
# ============================================================================


class TestQuantileMapping(unittest.TestCase):
    """Unit tests for the QM quantile mapping bias corrector."""

    def test_initialization_with_default_values(self):
        """Test QM initialization with default distributions."""

        model = QM()

        # By default, the number of features is inferred during fit.
        self.assertIsNone(model.n_features)

        # Verify the default clipping tolerance and empirical distributions.
        self.assertEqual(model._tol, 1e-3)
        self.assertIs(model._distY0.dist, rv_histogram)
        self.assertIs(model._distX0.dist, rv_histogram)

    def test_initialization_with_custom_tolerance(self):
        """Test QM initialization with a custom numerical tolerance."""

        model = QM(tol=1e-5)

        self.assertEqual(model._tol, 1e-5)

    def test_fit_infers_one_feature_from_1d_arrays(self):
        """Test that fit infers one feature from one-dimensional arrays."""

        # Build deterministic source and target samples linked by a linear rule.
        X = np.linspace(-3.0, 3.0, 201)
        Y = 2.0 * X + 5.0

        model = QM(
            distY0=sc.norm,
            distX0=sc.norm,
        )

        model.fit(Y, X)

        # One-dimensional arrays must be reshaped internally as one feature.
        self.assertEqual(model.n_features, 1)
        self.assertEqual(len(model._distY0.law), 1)
        self.assertEqual(len(model._distX0.law), 1)

    def test_fit_infers_multiple_features(self):
        """Test that fit infers the number of columns as features."""

        base = np.linspace(-3.0, 3.0, 201)

        # Create a two-feature biased dataset.
        X = np.column_stack(
            (
                base,
                2.0 * base,
            )
        )

        # Create a two-feature reference dataset with different transformations.
        Y = np.column_stack(
            (
                base + 5.0,
                0.5 * base - 3.0,
            )
        )

        model = QM(
            distY0=sc.norm,
            distX0=sc.norm,
        )

        model.fit(Y, X)

        # A separate distribution must be fitted for each feature.
        self.assertEqual(model.n_features, 2)
        self.assertEqual(len(model._distY0.law), 2)
        self.assertEqual(len(model._distX0.law), 2)

    def test_normal_quantile_mapping(self):
        """Test parametric quantile mapping between normal distributions."""

        X = np.linspace(-4.0, 4.0, 1001)

        # The target data follows a known linear location-scale transformation.
        Y = 10.0 + 2.0 * X

        model = QM(
            distY0=sc.norm,
            distX0=sc.norm,
        )

        model.fit(Y, X)

        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        corrected = model.predict(values)

        # Normal-to-normal quantile mapping should reproduce the same
        # location-scale transformation used to generate Y.
        expected = 10.0 + 2.0 * values

        # One-dimensional inputs are returned as a two-dimensional column.
        self.assertEqual(corrected.shape, (values.size, 1))

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_quantile_mapping_with_frozen_distributions(self):
        """Test quantile mapping using frozen source and target laws."""

        # Frozen distributions already contain their fitted parameters,
        # so no training samples are required during fit.
        model = QM(
            n_features=1,
            distX0=sc.norm(loc=0.0, scale=1.0),
            distY0=sc.norm(loc=10.0, scale=2.0),
        )

        model.fit(Y0=None, X0=None)

        values = np.array([-1.0, 0.0, 1.0])
        corrected = model.predict(values)

        # Standard-normal quantiles mapped to N(10, 2) give 10 + 2x.
        expected = np.array([8.0, 10.0, 12.0])

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_quantile_mapping_with_multiple_frozen_distributions(self):
        """Test feature-specific frozen distributions."""

        # Each feature uses its own source and target distribution.
        model = QM(
            n_features=2,
            distX0=[
                sc.norm(loc=0.0, scale=1.0),
                sc.norm(loc=10.0, scale=2.0),
            ],
            distY0=[
                sc.norm(loc=5.0, scale=2.0),
                sc.norm(loc=-3.0, scale=4.0),
            ],
        )

        model.fit(Y0=None, X0=None)

        values = np.array(
            [
                [-1.0, 8.0],
                [0.0, 10.0],
                [1.0, 12.0],
            ]
        )

        corrected = model.predict(values)

        # Expected values are computed from the location-scale mapping
        # associated with each pair of frozen normal distributions.
        expected = np.array(
            [
                [3.0, -7.0],
                [5.0, -3.0],
                [7.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            corrected,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_empirical_quantile_mapping(self):
        """Test quantile mapping with the default empirical distributions."""

        X = np.linspace(0.0, 10.0, 501)

        # Use an exact monotonic relationship between source and target data.
        Y = 3.0 * X + 2.0

        # The default QM implementation uses empirical rv_histogram laws.
        model = QM()
        model.fit(Y, X)

        values = np.array([2.0, 4.0, 6.0, 8.0])
        corrected = model.predict(values)

        expected = 3.0 * values + 2.0

        # A wider tolerance is used because empirical interpolation introduces
        # a small discretization error.
        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            atol=0.1,
        )

    def test_fit_with_frozen_target_and_fitted_source(self):
        """Test fitting only the source distribution."""

        X = np.linspace(-4.0, 4.0, 1001)

        # The source distribution must be estimated from X, while the target
        # distribution is already frozen with known parameters.
        model = QM(
            distX0=sc.norm,
            distY0=sc.norm(loc=20.0, scale=3.0),
        )

        model.fit(Y0=None, X0=X)

        values = np.array([-1.0, 0.0, 1.0])
        corrected = model.predict(values)

        # Reproduce the expected result manually using scipy:
        # 1. Fit the source normal distribution.
        # 2. Convert source values to probabilities.
        # 3. Convert probabilities through the frozen target distribution.
        fitted_location, fitted_scale = sc.norm.fit(X)

        probabilities = sc.norm(
            loc=fitted_location,
            scale=fitted_scale,
        ).cdf(values)

        expected = sc.norm(
            loc=20.0,
            scale=3.0,
        ).ppf(probabilities)

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


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
