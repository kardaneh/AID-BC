# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import xarray as xr


class ClimateDataset:
    """
    Climate dataset preprocessing pipeline.

    This class:
    loads ERA5 and CMIP6 dataset, harmonizes coordinates,
    adds cyclic longitude, interpolates CMIP6 onto ERA5 grid
    and exposes aligned DataArrays for downstream processing

    Attributes
    ----------
    era5_data : xarray.DataArray
        Reference ERA5 variable.

    cmip6_data : xarray.DataArray
        Interpolated CMIP6 variable on ERA5 grid.
    """

    def __init__(self, era5_path, cmip6_path, variable_name, logger=None):
        """
        Initialize the climate dataset handler.

        Parameters
        ----------
        era5_path : str
            Path to ERA5 NetCDF file.

        cmip6_path : str
            Path to CMIP6 NetCDF file.

        variable_name : str
            Name of the climate variable to process.

        logger : Logger, optional
            Custom logger instance.
        """

        self.era5_path = era5_path
        self.cmip6_path = cmip6_path

        self.variable_name = variable_name

        self.logger = logger

        self.era5 = None
        self.cmip6 = None

        self.era5_data = None
        self.cmip6_data = None

    def load(self):
        """
        Load ERA5 and CMIP6 datasets.
        """

        if self.logger:
            self.logger.info("Loading ERA5 dataset")

        self.era5 = xr.open_dataset(self.era5_path)

        if self.logger:
            self.logger.info("Loading CMIP6 dataset")

        self.cmip6 = xr.open_dataset(self.cmip6_path)

        # Check variable existence
        if self.variable_name not in self.cmip6:
            raise ValueError(
                f"Variable '{self.variable_name}' " "not found in CMIP6 dataset"
            )

        if self.variable_name not in self.era5:
            raise ValueError(
                f"Variable '{self.variable_name}' " "not found in ERA5 dataset"
            )

        if self.logger:
            self.logger.success("Datasets loaded successfully")

    def rename_cmip6_coordinates(self):
        """
        Rename CMIP6 coordinates to match ERA5 convention.
        """

        rename_dict = {}

        # Rename CMIP6 latitude coordinate if it uses the name "lat"
        if "lat" in self.cmip6.coords:
            rename_dict["lat"] = "latitude"

        # Rename CMIP6 longitude coordinate if it uses the name "lon"
        if "lon" in self.cmip6.coords:
            rename_dict["lon"] = "longitude"

        if rename_dict:
            if self.logger:
                self.logger.info(f"Renaming CMIP6 coordinates: {rename_dict}")

            # Rename CMIP6 coordinates to match ERA5 coordinate names
            self.cmip6 = self.cmip6.rename(rename_dict)

    @staticmethod
    def make_longitude_cyclic(da):
        """
        Add cyclic longitude point to avoid interpolation
        artifacts at the dateline.

        Parameters
        ----------
        da : xarray.DataArray
            Input data array with a longitude coordinate.

        Returns
        -------
        da_ext : xarray.DataArray
            Data array extended with one additional cyclic longitude point.
        """

        # Append the first longitude slice to the end of the data array,
        # This closes the longitude cycle and helps avoid edge effects.
        da_ext = xr.concat([da, da.isel(longitude=0)], dim="longitude")

        # Create the extended longitude coordinate by adding 360 degrees
        # to the first longitude value.
        new_lon = np.append(da["longitude"].values, da["longitude"].values[0] + 360)

        # Assign the updated longitude coordinate to the extended data array
        da_ext = da_ext.assign_coords(longitude=new_lon)

        return da_ext

    def interpolate_cmip6(self):
        """
        Interpolate CMIP6 variable onto ERA5 grid.
        """

        if self.logger:
            self.logger.info("Adding cyclic longitude to CMIP6 data")

        # Add a cyclic longitude point to the CMIP6 variable before interpolation
        cmip6_fixed = self.make_longitude_cyclic(self.cmip6[self.variable_name])

        if self.logger:
            self.logger.info("Interpolating CMIP6 onto ERA5 grid")

        # Interpolate CMIP6 data onto the latitude and longitude grid of ERA5
        self.cmip6_data = cmip6_fixed.interp(
            latitude=self.era5["latitude"],
            longitude=self.era5["longitude"],
            method="linear",
        )

        # Store the ERA5 reference variable without interpolation
        self.era5_data = self.era5[self.variable_name]

        if self.logger:
            self.logger.success(f"Interpolation completed: " f"{self.cmip6_data.shape}")

    def prepare(self):
        """
        Run the complete preprocessing pipeline.
        """

        # Load ERA5 and CMIP6 datasets
        self.load()

        # Rename CMIP6 coordinates to match ERA5 coordinate names
        self.rename_cmip6_coordinates()

        # Interpolate CMIP6 data onto the ERA5 grid
        self.interpolate_cmip6()

        if self.logger:
            self.logger.success("Dataset preparation completed")
