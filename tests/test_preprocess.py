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
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.preprocess import build_paths, preprocess_year, main


# python tests/test_preprocess.py


# ============================================================================
# Test Utilities
# ============================================================================


def create_yearly_dummy_era5_netcdf(
    root_dir,
    year,
    variable_name="tas",
    n_time=4,
):
    """Create a yearly dummy ERA5 NetCDF file."""

    os.makedirs(root_dir, exist_ok=True)

    latitude = np.linspace(-90, 90, 10)
    longitude = np.linspace(0, 350, 20)
    time = pd.date_range(f"{year}-01-01", periods=n_time, freq="6h")

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

    path = os.path.join(root_dir, f"samples_{year}.nc")
    ds.to_netcdf(path)

    return path, ds


def create_yearly_dummy_cmip6_netcdf(
    root_dir,
    year,
    variable_name="tas",
    n_time=4,
    use_lat_lon=True,
):
    """Create a yearly dummy CMIP6 NetCDF file."""

    os.makedirs(root_dir, exist_ok=True)

    lat_name = "lat" if use_lat_lon else "latitude"
    lon_name = "lon" if use_lat_lon else "longitude"

    latitude = np.linspace(-90, 90, 6)
    longitude = np.linspace(0, 300, 12)
    time = pd.date_range(f"{year}-01-01", periods=n_time, freq="6h")

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

    path = os.path.join(root_dir, f"samples_{year}.nc")
    ds.to_netcdf(path)

    return path, ds


# ============================================================================
# Unit Tests for preprocess.py
# ============================================================================


class TestPreprocess(unittest.TestCase):
    """Unit tests for preprocess."""

    def setUp(self):
        """Set up test fixtures."""

        self.temp_dir = tempfile.mkdtemp()

        self.era5_root = os.path.join(
            self.temp_dir,
            "era5",
        )

        self.cmip6_root = os.path.join(
            self.temp_dir,
            "cmip6",
        )

        self.output_zarr = os.path.join(
            self.temp_dir,
            "output",
            "cmip6_test.zarr",
        )

        self.variable_name = "tas"

        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup - created temp directory: {self.temp_dir}")

    # ------------------------------------------------------------------------
    # Path Tests
    # ------------------------------------------------------------------------

    def test_build_paths(self):
        """Test yearly ERA5 and CMIP6 path construction."""

        year = 2020

        era5_path, cmip6_path = build_paths(
            year=year,
            era5_root=self.era5_root,
            cmip6_root=self.cmip6_root,
        )

        self.assertEqual(
            era5_path,
            os.path.join(self.era5_root, "samples_2020.nc"),
        )

        self.assertEqual(
            cmip6_path,
            os.path.join(self.cmip6_root, "samples_2020.nc"),
        )

        if self.logger:
            self.logger.info("✅ build_paths test passed")

    # ------------------------------------------------------------------------
    # preprocess_year Tests
    # ------------------------------------------------------------------------

    def test_preprocess_year_returns_interpolated_dataarray(self):
        """Test preprocessing one CMIP6 year onto the ERA5 grid."""

        year = 2020

        _, era5_ds = create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
        )

        create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
            use_lat_lon=True,
        )

        da = preprocess_year(
            year=year,
            era5_root=self.era5_root,
            cmip6_root=self.cmip6_root,
            variable_name=self.variable_name,
            logger=self.logger,
        )

        self.assertEqual(
            da.name,
            self.variable_name,
        )

        self.assertEqual(
            da.dtype,
            np.float32,
        )

        self.assertEqual(
            da.dims,
            ("time", "latitude", "longitude"),
        )

        self.assertEqual(
            da.sizes["time"],
            era5_ds.sizes["time"],
        )

        self.assertEqual(
            da.sizes["latitude"],
            era5_ds.sizes["latitude"],
        )

        self.assertEqual(
            da.sizes["longitude"],
            era5_ds.sizes["longitude"],
        )

        np.testing.assert_allclose(
            da["latitude"].values,
            era5_ds["latitude"].values,
        )

        np.testing.assert_allclose(
            da["longitude"].values,
            era5_ds["longitude"].values,
        )

        if self.logger:
            self.logger.info(f"✅ preprocess_year test passed - shape: {da.shape}")

    # ------------------------------------------------------------------------
    # main() Tests
    # ------------------------------------------------------------------------

    def test_main_writes_single_year_zarr(self):
        """Test that main writes one preprocessed year to a Zarr store."""

        year = 2020

        create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
        )

        create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
            use_lat_lon=True,
        )

        test_args = [
            "preprocess.py",
            "--start_year",
            str(year),
            "--end_year",
            str(year),
            "--variable",
            self.variable_name,
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--output_zarr",
            self.output_zarr,
            "--time_chunk",
            "4",
            "--lat_chunk",
            "5",
            "--lon_chunk",
            "10",
            "--overwrite",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        self.assertTrue(os.path.exists(self.output_zarr))

        ds = xr.open_zarr(self.output_zarr)

        self.assertIn(
            self.variable_name,
            ds.data_vars,
        )

        da = ds[self.variable_name]

        self.assertEqual(
            da.shape,
            (4, 10, 20),
        )

        self.assertEqual(
            da.dtype,
            np.float32,
        )

        self.assertEqual(
            da.dims,
            ("time", "latitude", "longitude"),
        )

        if self.logger:
            self.logger.info("✅ main single-year Zarr test passed")

    def test_main_appends_multiple_years_to_zarr(self):
        """Test that main appends multiple years along the time dimension."""

        years = [2020, 2021]

        for year in years:
            create_yearly_dummy_era5_netcdf(
                root_dir=self.era5_root,
                year=year,
                variable_name=self.variable_name,
                n_time=4,
            )

            create_yearly_dummy_cmip6_netcdf(
                root_dir=self.cmip6_root,
                year=year,
                variable_name=self.variable_name,
                n_time=4,
                use_lat_lon=True,
            )

        test_args = [
            "preprocess.py",
            "--start_year",
            "2020",
            "--end_year",
            "2021",
            "--variable",
            self.variable_name,
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--output_zarr",
            self.output_zarr,
            "--time_chunk",
            "4",
            "--lat_chunk",
            "5",
            "--lon_chunk",
            "10",
            "--overwrite",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        ds = xr.open_zarr(self.output_zarr)

        da = ds[self.variable_name]

        self.assertEqual(
            da.shape,
            (8, 10, 20),
        )

        self.assertEqual(
            da.sizes["time"],
            8,
        )

        self.assertEqual(
            str(da["time"].values[0])[:10],
            "2020-01-01",
        )

        self.assertEqual(
            str(da["time"].values[-1])[:10],
            "2021-01-01",
        )

        if self.logger:
            self.logger.info("✅ main multi-year append Zarr test passed")

    def test_main_raises_if_zarr_exists_without_overwrite(self):
        """Test that main raises FileExistsError when output exists without overwrite."""

        year = 2020

        create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
        )

        create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
            use_lat_lon=True,
        )

        os.makedirs(
            self.output_zarr,
            exist_ok=True,
        )

        test_args = [
            "preprocess.py",
            "--start_year",
            str(year),
            "--end_year",
            str(year),
            "--variable",
            self.variable_name,
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--output_zarr",
            self.output_zarr,
            "--time_chunk",
            "4",
            "--lat_chunk",
            "5",
            "--lon_chunk",
            "10",
        ]

        with patch.object(sys, "argv", test_args):
            with self.assertRaises(FileExistsError):
                main()

        if self.logger:
            self.logger.info("✅ overwrite protection test passed")

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def tearDown(self):
        """Clean up after tests."""

        shutil.rmtree(self.temp_dir)

        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all preprocess tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPreprocess))

    runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
