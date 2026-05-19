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
from AID_BC.dataset import ClimateDataset

# python tests/test_dataset.py


# ============================================================================
# Test Utilities
# ============================================================================


def create_dummy_era5_netcdf(temp_dir, variable_name="tas"):
    """Create a dummy ERA5 NetCDF file for testing."""

    latitude = np.linspace(-90, 90, 10)
    longitude = np.linspace(0, 350, 20)
    time = pd.date_range("2020-01-01", periods=4, freq="6h")

    data = np.random.randn(
        len(time),
        len(latitude),
        len(longitude),
    ).astype(np.float32)

    ds = xr.Dataset(
        {
            variable_name: (
                ("time", "latitude", "longitude"),
                data,
            )
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    path = os.path.join(temp_dir, "era5.nc")
    ds.to_netcdf(path)

    return path, ds


def create_dummy_cmip6_netcdf(
    temp_dir,
    variable_name="tas",
    use_lat_lon=True,
):
    """Create a dummy CMIP6 NetCDF file for testing."""

    lat_name = "lat" if use_lat_lon else "latitude"
    lon_name = "lon" if use_lat_lon else "longitude"

    latitude = np.linspace(-90, 90, 6)
    longitude = np.linspace(0, 300, 12)
    time = pd.date_range("2020-01-01", periods=4, freq="6h")

    data = np.random.randn(
        len(time),
        len(latitude),
        len(longitude),
    ).astype(np.float32)

    ds = xr.Dataset(
        {
            variable_name: (
                ("time", lat_name, lon_name),
                data,
            )
        },
        coords={
            "time": time,
            lat_name: latitude,
            lon_name: longitude,
        },
    )

    path = os.path.join(temp_dir, "cmip6.nc")
    ds.to_netcdf(path)

    return path, ds


def create_dummy_dataset_without_variable(
    temp_dir,
    filename,
    variable_name="wrong_var",
    coord_names=("latitude", "longitude"),
):
    """Create a dummy NetCDF file without the expected variable."""

    lat_name, lon_name = coord_names

    latitude = np.linspace(-90, 90, 5)
    longitude = np.linspace(0, 300, 8)
    time = pd.date_range("2020-01-01", periods=2, freq="6h")

    data = np.random.randn(
        len(time),
        len(latitude),
        len(longitude),
    ).astype(np.float32)

    ds = xr.Dataset(
        {
            variable_name: (
                ("time", lat_name, lon_name),
                data,
            )
        },
        coords={
            "time": time,
            lat_name: latitude,
            lon_name: longitude,
        },
    )

    path = os.path.join(temp_dir, filename)
    ds.to_netcdf(path)

    return path, ds


# ============================================================================
# Unit Tests for ClimateDataset
# ============================================================================


class TestClimateDataset(unittest.TestCase):
    """Unit tests for ClimateDataset class."""

    def setUp(self):
        """Set up test fixtures."""

        self.temp_dir = tempfile.mkdtemp()

        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup - created temp directory: {self.temp_dir}")

        self.variable_name = "tas"

        self.era5_path, self.era5_ds = create_dummy_era5_netcdf(
            self.temp_dir,
            variable_name=self.variable_name,
        )

        self.cmip6_path, self.cmip6_ds = create_dummy_cmip6_netcdf(
            self.temp_dir,
            variable_name=self.variable_name,
            use_lat_lon=True,
        )

        if self.logger:
            self.logger.info("Test setup complete - ClimateDataset test fixtures ready")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_climate_dataset_initialization(self):
        """Test ClimateDataset initialization."""

        if self.logger:
            self.logger.info("Testing ClimateDataset initialization")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        self.assertEqual(dataset.era5_path, self.era5_path)
        self.assertEqual(dataset.cmip6_path, self.cmip6_path)
        self.assertEqual(dataset.variable_name, self.variable_name)
        self.assertIsNone(dataset.era5)
        self.assertIsNone(dataset.cmip6)
        self.assertIsNone(dataset.era5_data)
        self.assertIsNone(dataset.cmip6_data)

        if self.logger:
            self.logger.info("✅ ClimateDataset initialization test passed")

    # ------------------------------------------------------------------------
    # Load Method Tests
    # ------------------------------------------------------------------------

    def test_load_datasets_successfully(self):
        """Test loading ERA5 and CMIP6 datasets."""

        if self.logger:
            self.logger.info("Testing dataset loading")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.load()

        self.assertIsNotNone(dataset.era5)
        self.assertIsNotNone(dataset.cmip6)

        self.assertIn(self.variable_name, dataset.era5)
        self.assertIn(self.variable_name, dataset.cmip6)

        self.assertEqual(
            dataset.era5[self.variable_name].shape,
            self.era5_ds[self.variable_name].shape,
        )

        self.assertEqual(
            dataset.cmip6[self.variable_name].shape,
            self.cmip6_ds[self.variable_name].shape,
        )

        if self.logger:
            self.logger.info("✅ Dataset loading test passed")

    def test_load_raises_error_if_variable_missing_in_era5(self):
        """Test load raises ValueError when variable is missing in ERA5."""

        if self.logger:
            self.logger.info("Testing missing variable in ERA5")

        bad_era5_path, _ = create_dummy_dataset_without_variable(
            self.temp_dir,
            filename="bad_era5.nc",
            variable_name="wrong_var",
            coord_names=("latitude", "longitude"),
        )

        dataset = ClimateDataset(
            era5_path=bad_era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        with self.assertRaises(ValueError) as context:
            dataset.load()

        self.assertIn(
            f"Variable '{self.variable_name}' not found in ERA5 dataset",
            str(context.exception),
        )

        if self.logger:
            self.logger.info("✅ Missing variable in ERA5 test passed")

    def test_load_raises_error_if_variable_missing_in_cmip6(self):
        """Test load raises ValueError when variable is missing in CMIP6."""

        if self.logger:
            self.logger.info("Testing missing variable in CMIP6")

        bad_cmip6_path, _ = create_dummy_dataset_without_variable(
            self.temp_dir,
            filename="bad_cmip6.nc",
            variable_name="wrong_var",
            coord_names=("lat", "lon"),
        )

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=bad_cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        with self.assertRaises(ValueError) as context:
            dataset.load()

        self.assertIn(
            f"Variable '{self.variable_name}' not found in CMIP6 dataset",
            str(context.exception),
        )

        if self.logger:
            self.logger.info("✅ Missing variable in CMIP6 test passed")

    # ------------------------------------------------------------------------
    # Coordinate Renaming Tests
    # ------------------------------------------------------------------------

    def test_rename_cmip6_coordinates_from_lat_lon(self):
        """Test CMIP6 coordinate renaming from lat/lon to latitude/longitude."""

        if self.logger:
            self.logger.info("Testing CMIP6 coordinate renaming")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.load()

        self.assertIn("lat", dataset.cmip6.coords)
        self.assertIn("lon", dataset.cmip6.coords)

        dataset.rename_cmip6_coordinates()

        self.assertIn("latitude", dataset.cmip6.coords)
        self.assertIn("longitude", dataset.cmip6.coords)
        self.assertNotIn("lat", dataset.cmip6.coords)
        self.assertNotIn("lon", dataset.cmip6.coords)

        if self.logger:
            self.logger.info("✅ CMIP6 coordinate renaming test passed")

    def test_rename_cmip6_coordinates_when_already_standard(self):
        """Test coordinate renaming when CMIP6 coordinates already match ERA5."""

        if self.logger:
            self.logger.info("Testing CMIP6 coordinates already standardized")

        cmip6_standard_path, _ = create_dummy_cmip6_netcdf(
            self.temp_dir,
            variable_name=self.variable_name,
            use_lat_lon=False,
        )

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=cmip6_standard_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.load()
        dataset.rename_cmip6_coordinates()

        self.assertIn("latitude", dataset.cmip6.coords)
        self.assertIn("longitude", dataset.cmip6.coords)

        if self.logger:
            self.logger.info("✅ Already standardized coordinates test passed")

    # ------------------------------------------------------------------------
    # Longitude Cyclic Tests
    # ------------------------------------------------------------------------

    def test_make_longitude_cyclic(self):
        """Test that cyclic longitude point is added correctly."""

        if self.logger:
            self.logger.info("Testing make_longitude_cyclic")

        latitude = np.linspace(-90, 90, 4)
        longitude = np.array([0, 90, 180, 270])

        values = np.random.randn(
            len(latitude),
            len(longitude),
        ).astype(np.float32)

        da = xr.DataArray(
            values,
            dims=("latitude", "longitude"),
            coords={
                "latitude": latitude,
                "longitude": longitude,
            },
        )

        da_ext = ClimateDataset.make_longitude_cyclic(da)

        self.assertEqual(
            da_ext.sizes["longitude"],
            da.sizes["longitude"] + 1,
        )

        self.assertEqual(
            da_ext["longitude"].values[-1],
            da["longitude"].values[0] + 360,
        )

        np.testing.assert_allclose(
            da_ext.isel(longitude=-1).values,
            da.isel(longitude=0).values,
        )

        if self.logger:
            self.logger.info("✅ make_longitude_cyclic test passed")

    # ------------------------------------------------------------------------
    # Interpolation Tests
    # ------------------------------------------------------------------------

    def test_interpolate_cmip6(self):
        """Test CMIP6 interpolation onto ERA5 grid."""

        if self.logger:
            self.logger.info("Testing CMIP6 interpolation onto ERA5 grid")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.load()
        dataset.rename_cmip6_coordinates()
        dataset.interpolate_cmip6()

        self.assertIsNotNone(dataset.era5_data)
        self.assertIsNotNone(dataset.cmip6_data)

        self.assertEqual(
            dataset.era5_data.sizes["latitude"],
            self.era5_ds.sizes["latitude"],
        )

        self.assertEqual(
            dataset.era5_data.sizes["longitude"],
            self.era5_ds.sizes["longitude"],
        )

        self.assertEqual(
            dataset.cmip6_data.sizes["latitude"],
            self.era5_ds.sizes["latitude"],
        )

        self.assertEqual(
            dataset.cmip6_data.sizes["longitude"],
            self.era5_ds.sizes["longitude"],
        )

        self.assertEqual(
            dataset.cmip6_data.shape,
            dataset.era5_data.shape,
        )

        if self.logger:
            self.logger.info(
                f"✅ CMIP6 interpolation test passed - shape: "
                f"{dataset.cmip6_data.shape}"
            )

    def test_interpolate_cmip6_matches_era5_coordinates(self):
        """Test interpolated CMIP6 data uses ERA5 coordinates."""

        if self.logger:
            self.logger.info("Testing interpolated coordinate alignment")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.load()
        dataset.rename_cmip6_coordinates()
        dataset.interpolate_cmip6()

        np.testing.assert_allclose(
            dataset.cmip6_data["latitude"].values,
            dataset.era5["latitude"].values,
        )

        np.testing.assert_allclose(
            dataset.cmip6_data["longitude"].values,
            dataset.era5["longitude"].values,
        )

        if self.logger:
            self.logger.info("✅ Interpolated coordinate alignment test passed")

    # ------------------------------------------------------------------------
    # Full Pipeline Tests
    # ------------------------------------------------------------------------

    def test_prepare_pipeline(self):
        """Test complete ClimateDataset preparation pipeline."""

        if self.logger:
            self.logger.info("Testing full prepare pipeline")

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        dataset.prepare()

        self.assertIsNotNone(dataset.era5)
        self.assertIsNotNone(dataset.cmip6)
        self.assertIsNotNone(dataset.era5_data)
        self.assertIsNotNone(dataset.cmip6_data)

        self.assertIn("latitude", dataset.cmip6.coords)
        self.assertIn("longitude", dataset.cmip6.coords)

        self.assertEqual(
            dataset.cmip6_data.shape,
            dataset.era5_data.shape,
        )

        if self.logger:
            self.logger.info("✅ Full prepare pipeline test passed")

    def test_prepare_pipeline_without_logger(self):
        """Test complete pipeline when logger is None."""

        dataset = ClimateDataset(
            era5_path=self.era5_path,
            cmip6_path=self.cmip6_path,
            variable_name=self.variable_name,
            logger=None,
        )

        dataset.prepare()

        self.assertIsNotNone(dataset.era5_data)
        self.assertIsNotNone(dataset.cmip6_data)

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def tearDown(self):
        """Clean up after tests."""

        shutil.rmtree(self.temp_dir)

        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all ClimateDataset tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestClimateDataset))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
