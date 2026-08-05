# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.dataset import ClimateDataset, resize_with_torchvision

# python -m unittest tests.test_dataset


# ============================================================================
# Test utilities
# ============================================================================


def create_dummy_era5_netcdf(
    temp_dir,
    variable_name="VAR_2T",
    latitude_descending=True,
    constant_value=None,
):
    """
    Create a dummy ERA5 NetCDF file.

    Parameters
    ----------
    temp_dir : str or pathlib.Path
        Directory in which the NetCDF file is created.
    variable_name : str, default="VAR_2T"
        Name of the climate variable.
    latitude_descending : bool, default=True
        Whether latitude is ordered north-to-south.
    constant_value : float or None, optional
        Constant value used for the field. If ``None``, deterministic random
        values are generated.

    Returns
    -------
    path : str
        Path to the generated NetCDF file.
    dataset : xarray.Dataset
        Dataset written to disk.
    """
    if latitude_descending:
        latitude = np.linspace(90.0, -90.0, 10)
    else:
        latitude = np.linspace(-90.0, 90.0, 10)

    # Include negative longitudes to test normalization to [0, 360).
    longitude = np.linspace(-180.0, 162.0, 20)
    time = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="6h",
    )

    if constant_value is None:
        rng = np.random.default_rng(10)
        data = rng.standard_normal(
            (
                len(time),
                len(latitude),
                len(longitude),
            )
        ).astype(np.float32)
    else:
        data = np.full(
            (
                len(time),
                len(latitude),
                len(longitude),
            ),
            constant_value,
            dtype=np.float32,
        )

    dataset = xr.Dataset(
        {
            variable_name: (
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
                data,
            )
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    path = os.path.join(
        temp_dir,
        "era5.nc",
    )
    dataset.to_netcdf(path)

    return path, dataset


def create_dummy_cmip6_netcdf(
    temp_dir,
    variable_name="VAR_2T",
    use_lat_lon=True,
    constant_value=None,
):
    """
    Create a dummy CMIP6 NetCDF file.

    Parameters
    ----------
    temp_dir : str or pathlib.Path
        Directory in which the NetCDF file is created.
    variable_name : str, default="VAR_2T"
        Name of the climate variable.
    use_lat_lon : bool, default=True
        Whether to use ``lat`` and ``lon`` instead of the standardized names.
    constant_value : float or None, optional
        Constant value used for the field. If ``None``, deterministic random
        values are generated.

    Returns
    -------
    path : str
        Path to the generated NetCDF file.
    dataset : xarray.Dataset
        Dataset written to disk.
    """
    latitude_name = "lat" if use_lat_lon else "latitude"
    longitude_name = "lon" if use_lat_lon else "longitude"

    latitude = np.linspace(-75.0, 75.0, 6)
    longitude = np.linspace(0.0, 330.0, 12)
    time = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="6h",
    )

    if constant_value is None:
        rng = np.random.default_rng(20)
        data = rng.standard_normal(
            (
                len(time),
                len(latitude),
                len(longitude),
            )
        ).astype(np.float32)
    else:
        data = np.full(
            (
                len(time),
                len(latitude),
                len(longitude),
            ),
            constant_value,
            dtype=np.float32,
        )

    dataset = xr.Dataset(
        {
            variable_name: (
                (
                    "time",
                    latitude_name,
                    longitude_name,
                ),
                data,
            )
        },
        coords={
            "time": time,
            latitude_name: latitude,
            longitude_name: longitude,
        },
    )

    path = os.path.join(
        temp_dir,
        "cmip6.nc",
    )
    dataset.to_netcdf(path)

    return path, dataset


def create_dataset_without_expected_variable(
    temp_dir,
    filename,
    coord_names,
):
    """
    Create a NetCDF file that does not contain the expected variable.

    Parameters
    ----------
    temp_dir : str or pathlib.Path
        Directory in which the file is created.
    filename : str
        NetCDF filename.
    coord_names : tuple of str
        Latitude and longitude coordinate names.

    Returns
    -------
    str
        Path to the generated NetCDF file.
    """
    latitude_name, longitude_name = coord_names

    dataset = xr.Dataset(
        {
            "wrong_variable": (
                (
                    "time",
                    latitude_name,
                    longitude_name,
                ),
                np.zeros(
                    (2, 4, 5),
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "time": pd.date_range(
                "2020-01-01",
                periods=2,
                freq="6h",
            ),
            latitude_name: np.linspace(
                -60.0,
                60.0,
                4,
            ),
            longitude_name: np.linspace(
                0.0,
                288.0,
                5,
            ),
        },
    )

    path = os.path.join(
        temp_dir,
        filename,
    )
    dataset.to_netcdf(path)

    return path


# ============================================================================
# Unit tests for resize_with_torchvision
# ============================================================================


class TestResizeWithTorchvision(unittest.TestCase):
    """Unit tests for the torchvision spatial resizing helper."""

    def test_resize_returns_requested_shape_and_float32(self):
        """Test output shape and dtype."""
        data = xr.DataArray(
            np.arange(
                2 * 4 * 6,
                dtype=np.float32,
            ).reshape(2, 4, 6),
            dims=(
                "time",
                "latitude",
                "longitude",
            ),
        )

        resized = resize_with_torchvision(
            data,
            target_shape=(3, 5),
        )

        self.assertEqual(
            resized.shape,
            (2, 3, 5),
        )
        self.assertEqual(
            resized.dtype,
            np.dtype("float32"),
        )

    def test_resize_preserves_constant_field(self):
        """Test that bilinear resizing preserves a constant field."""
        data = xr.DataArray(
            np.full(
                (2, 8, 12),
                280.0,
                dtype=np.float32,
            ),
            dims=(
                "time",
                "latitude",
                "longitude",
            ),
        )

        resized = resize_with_torchvision(
            data,
            target_shape=(5, 7),
        )

        np.testing.assert_allclose(
            resized,
            280.0,
            atol=1e-5,
        )


# ============================================================================
# Unit tests for ClimateDataset
# ============================================================================


class TestClimateDataset(unittest.TestCase):
    """Unit tests for the current ClimateDataset API."""

    def setUp(self):
        """Create temporary input files and a logger."""
        self.temp_dir = tempfile.mkdtemp()
        self.variable_name = "VAR_2T"

        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        (
            self.era5_path,
            self.era5_dataset,
        ) = create_dummy_era5_netcdf(
            self.temp_dir,
            variable_name=self.variable_name,
            latitude_descending=True,
        )

        (
            self.cmip6_path,
            self.cmip6_dataset,
        ) = create_dummy_cmip6_netcdf(
            self.temp_dir,
            variable_name=self.variable_name,
            use_lat_lon=True,
        )

    # ------------------------------------------------------------------------
    # Initialization and loading tests
    # ------------------------------------------------------------------------

    def test_initialization(self):
        """Test initialization of instance attributes."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            era5_path=self.era5_path,
            logger=self.logger,
        )

        self.assertEqual(
            dataset.era5_path,
            self.era5_path,
        )
        self.assertEqual(
            dataset.cmip6_path,
            self.cmip6_path,
        )
        self.assertEqual(
            dataset.variable_name,
            self.variable_name,
        )
        self.assertIsNone(dataset.era5)
        self.assertIsNone(dataset.cmip6)
        self.assertIsNone(dataset.era5_data)
        self.assertIsNone(dataset.cmip6_data)
        self.assertIsNone(dataset.era5_lat_descending)

    def test_load_era5_and_cmip6(self):
        """Test the two current loading methods."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            era5_path=self.era5_path,
        )

        dataset.load_era5()
        dataset.load_cmip6()

        self.assertIn(
            self.variable_name,
            dataset.era5,
        )
        self.assertIn(
            self.variable_name,
            dataset.cmip6,
        )

        dataset.close()

    def test_load_era5_requires_path(self):
        """Test rejection of ERA5 loading without an ERA5 path."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
        )

        with self.assertRaisesRegex(
            ValueError,
            "era5_path is required",
        ):
            dataset.load_era5()

    def test_load_era5_rejects_missing_variable(self):
        """Test rejection of an ERA5 file missing the requested variable."""
        bad_path = create_dataset_without_expected_variable(
            self.temp_dir,
            "bad_era5.nc",
            (
                "latitude",
                "longitude",
            ),
        )

        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            era5_path=bad_path,
        )

        with self.assertRaisesRegex(
            ValueError,
            "not found in ERA5 dataset",
        ):
            dataset.load_era5()

        dataset.close()

    def test_load_cmip6_rejects_missing_variable(self):
        """Test rejection of a CMIP6 file missing the requested variable."""
        bad_path = create_dataset_without_expected_variable(
            self.temp_dir,
            "bad_cmip6.nc",
            (
                "lat",
                "lon",
            ),
        )

        dataset = ClimateDataset(
            cmip6_path=bad_path,
            variable_name=self.variable_name,
        )

        with self.assertRaisesRegex(
            ValueError,
            "not found in CMIP6 dataset",
        ):
            dataset.load_cmip6()

        dataset.close()

    # ------------------------------------------------------------------------
    # Coordinate helper tests
    # ------------------------------------------------------------------------

    def test_rename_coordinates(self):
        """Test renaming from lat/lon to latitude/longitude."""
        renamed = ClimateDataset.rename_coordinates(self.cmip6_dataset)

        self.assertIn(
            "latitude",
            renamed.coords,
        )
        self.assertIn(
            "longitude",
            renamed.coords,
        )
        self.assertNotIn(
            "lat",
            renamed.coords,
        )
        self.assertNotIn(
            "lon",
            renamed.coords,
        )

    def test_is_latitude_descending(self):
        """Test detection of latitude orientation."""
        descending = xr.Dataset(
            coords={
                "latitude": [60.0, 0.0, -60.0],
            }
        )
        ascending = xr.Dataset(
            coords={
                "latitude": [-60.0, 0.0, 60.0],
            }
        )

        self.assertTrue(ClimateDataset.is_latitude_descending(descending))
        self.assertFalse(ClimateDataset.is_latitude_descending(ascending))

    def test_normalize_longitudes_sorts_and_removes_duplicates(self):
        """Test longitude normalization and duplicate removal."""
        dataset = xr.Dataset(
            {
                self.variable_name: (
                    (
                        "latitude",
                        "longitude",
                    ),
                    np.arange(
                        10,
                        dtype=np.float32,
                    ).reshape(2, 5),
                )
            },
            coords={
                "latitude": [-30.0, 30.0],
                "longitude": [
                    -180.0,
                    0.0,
                    180.0,
                    360.0,
                    90.0,
                ],
            },
        )

        normalized = ClimateDataset.normalize_longitudes(dataset)

        longitude = normalized["longitude"].values

        self.assertTrue(np.all(longitude >= 0.0))
        self.assertTrue(np.all(longitude < 360.0))
        self.assertTrue(np.all(np.diff(longitude) > 0.0))
        self.assertEqual(
            len(longitude),
            len(np.unique(longitude)),
        )
        np.testing.assert_allclose(
            longitude,
            np.array([0.0, 90.0, 180.0]),
        )

    def test_sort_latitude(self):
        """Test ascending and descending latitude sorting."""
        dataset = xr.Dataset(
            coords={
                "latitude": [
                    0.0,
                    -30.0,
                    30.0,
                ],
                "longitude": [0.0],
            }
        )

        ascending = ClimateDataset.sort_latitude(
            dataset,
            descending=False,
        )
        descending = ClimateDataset.sort_latitude(
            dataset,
            descending=True,
        )

        np.testing.assert_allclose(
            ascending["latitude"].values,
            [-30.0, 0.0, 30.0],
        )
        np.testing.assert_allclose(
            descending["latitude"].values,
            [30.0, 0.0, -30.0],
        )

    def test_standardize_coordinates(self):
        """Test coordinate renaming, longitude normalization, and sorting."""
        standardized = ClimateDataset.standardize_coordinates(
            self.cmip6_dataset,
            latitude_descending=True,
        )

        self.assertIn(
            "latitude",
            standardized.coords,
        )
        self.assertIn(
            "longitude",
            standardized.coords,
        )
        self.assertGreater(
            standardized["latitude"].values[0],
            standardized["latitude"].values[-1],
        )
        self.assertTrue(np.all(standardized["longitude"].values >= 0.0))
        self.assertTrue(np.all(standardized["longitude"].values < 360.0))

    # ------------------------------------------------------------------------
    # Full preparation pipeline tests
    # ------------------------------------------------------------------------

    def test_prepare_dataset_with_era5(self):
        """Test ERA5 resizing onto the native CMIP6 grid."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            era5_path=self.era5_path,
            logger=self.logger,
        )

        (
            era5_on_cmip6,
            cmip6_native,
            latitude_descending,
        ) = dataset.prepare_dataset()

        self.assertTrue(latitude_descending)

        self.assertEqual(
            era5_on_cmip6.dims,
            (
                "time",
                "latitude",
                "longitude",
            ),
        )
        self.assertEqual(
            cmip6_native.dims,
            (
                "time",
                "latitude",
                "longitude",
            ),
        )

        # Both arrays must use the native CMIP6 spatial grid.
        self.assertEqual(
            era5_on_cmip6.shape,
            cmip6_native.shape,
        )
        self.assertEqual(
            era5_on_cmip6.sizes["latitude"],
            self.cmip6_dataset.sizes["lat"],
        )
        self.assertEqual(
            era5_on_cmip6.sizes["longitude"],
            self.cmip6_dataset.sizes["lon"],
        )

        np.testing.assert_allclose(
            era5_on_cmip6["latitude"].values,
            cmip6_native["latitude"].values,
        )
        np.testing.assert_allclose(
            era5_on_cmip6["longitude"].values,
            cmip6_native["longitude"].values,
        )
        np.testing.assert_array_equal(
            era5_on_cmip6["time"].values,
            self.era5_dataset["time"].values,
        )

        self.assertGreater(
            era5_on_cmip6["latitude"].values[0],
            era5_on_cmip6["latitude"].values[-1],
        )

        self.assertIs(
            dataset.era5_data,
            era5_on_cmip6,
        )
        self.assertIs(
            dataset.cmip6_data,
            cmip6_native,
        )

        dataset.close()

    def test_prepare_dataset_without_era5_requires_orientation(self):
        """Test that CMIP6-only mode requires latitude orientation."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
        )

        with self.assertRaisesRegex(
            ValueError,
            "latitude_descending is required",
        ):
            dataset.prepare_dataset()

    def test_prepare_dataset_without_logger(self):
        """Test the full ERA5-CMIP6 mode without a logger."""
        dataset = ClimateDataset(
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            era5_path=self.era5_path,
            logger=None,
        )

        era5_on_cmip6, cmip6_native, _ = dataset.prepare_dataset()

        self.assertEqual(
            era5_on_cmip6.shape,
            cmip6_native.shape,
        )

        dataset.close()

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def tearDown(self):
        """Clean up after tests."""

        # Close the datasets returned by helper functions before deleting files.
        self.era5_dataset.close()
        self.cmip6_dataset.close()
        shutil.rmtree(self.temp_dir)

        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all ClimateDataset tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestResizeWithTorchvision))
    suite.addTests(loader.loadTestsFromTestCase(TestClimateDataset))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
