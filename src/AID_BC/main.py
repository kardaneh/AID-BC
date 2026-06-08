# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import gc
from pathlib import Path

import numpy as np
import xarray as xr

from AID_BC.logger import Logger
from AID_BC.quantile_mapping import QM


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Quantile Mapping bias correction using preprocessed Zarr data"
    )

    # First year used to train the QM correction
    parser.add_argument(
        "--train_start", type=int, required=True, help="Training start year"
    )

    # Last year used to train the QM correction
    parser.add_argument(
        "--train_end", type=int, required=True, help="Training end year"
    )

    parser.add_argument(
        "--apply_start", type=int, required=True, help="Application start year"
    )

    parser.add_argument(
        "--apply_end", type=int, required=True, help="Application end year"
    )

    parser.add_argument("--variable", type=str, default="VAR_2T", help="Variable name")

    parser.add_argument(
        "--era5_root", type=str, required=True, help="ERA5 root directory"
    )

    parser.add_argument(
        "--cmip6_train_zarr",
        type=str,
        required=True,
        help="Preprocessed CMIP6 training Zarr path",
    )

    parser.add_argument(
        "--cmip6_apply_zarr",
        type=str,
        required=True,
        help="Preprocessed CMIP6 application Zarr path",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    # Number of latitude points processed at once
    parser.add_argument(
        "--chunk_lat", type=int, default=144, help="Latitude chunk size"
    )

    # Number of longitude points processed at once
    parser.add_argument(
        "--chunk_lon", type=int, default=360, help="Longitude chunk size"
    )

    return parser.parse_args()


def build_era5_paths(start_year, end_year, era5_root):
    """
    Build ERA5 file paths for the training period.

    Parameters
    ----------
    start_year : int
        Training start year.

    end_year : int
        Training end year.

    era5_root : str
        ERA5 root directory.

    Returns
    -------
    list[str]
        ERA5 file paths.
    """

    # ERA5 files are expected to follow the naming convention samples_<year>.nc
    # same as IPSL-AID
    return [
        str(Path(era5_root) / f"samples_{year}.nc")
        for year in range(start_year, end_year + 1)
    ]


def select_year_range(da, start_year, end_year):
    """
    Select a year range from a DataArray using the time coordinate.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time coordinate.

    start_year : int
        First year to select.

    end_year : int
        Last year to select.

    Returns
    -------
    xr.DataArray
        Selected data.
    """

    return da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def save_corrected_by_year(corr, output_dir, start_year, end_year, logger):
    """
    Save corrected data year by year as samples_<year>.nc.

    Parameters
    ----------
    corr : xr.DataArray
        Corrected application data.

    output_dir : str or Path
        Output directory.

    start_year : int
        First year to save.

    end_year : int
        Last year to save.

    logger : Logger
        Logger instance.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        logger.info(f"Saving corrected year {year}")

        corr_year = corr.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

        if corr_year.sizes["time"] == 0:
            raise ValueError(
                f"No data found for year {year} in corrected output. "
                f"Check the time coordinate of cmip6_apply_zarr."
            )

        output_file = output_dir / f"samples_{year}.nc"

        logger.info(f"Writing:\n{output_file}")

        corr_year.to_netcdf(output_file)


def iter_spatial_chunks(n_lat, n_lon, chunk_lat, chunk_lon):
    """
    Iterate over spatial chunks.

    Parameters
    ----------
    n_lat : int
        Number of latitude points.

    n_lon : int
        Number of longitude points.

    chunk_lat : int
        Latitude chunk size.

    chunk_lon : int
        Longitude chunk size.

    Returns
    ------
    tuple[slice, slice]
        Latitude and longitude slices defining one spatial chunk.
    """

    # Loop over latitude indices by chunk
    for lat_start in range(0, n_lat, chunk_lat):
        # Ensure the last latitude chunk does not exceed the grid size
        lat_end = min(lat_start + chunk_lat, n_lat)

        # Loop over longitude indices by chunk
        for lon_start in range(0, n_lon, chunk_lon):
            # Ensure the last longitude chunk does not exceed the grid size
            lon_end = min(lon_start + chunk_lon, n_lon)

            # Return slices defining the current spatial chunk
            yield (slice(lat_start, lat_end), slice(lon_start, lon_end))


def apply_qm_by_spatial_chunks(
    Y_train, X_train, X_apply, variable_name, chunk_lat, chunk_lon, logger
):
    """
    Apply Quantile Mapping chunk by chunk.

    Parameters
    ----------
    Y_train : xr.DataArray
        ERA5 reference training data.

    X_train : xr.DataArray
        Preprocessed CMIP6 training data on ERA5 grid.

    X_apply : xr.DataArray
        Preprocessed CMIP6 application data on ERA5 grid.

    variable_name : str
        Variable name.

    chunk_lat : int
        Latitude chunk size.

    chunk_lon : int
        Longitude chunk size.

    logger : Logger
        Logger instance.

    Returns
    -------
    corr : xr.DataArray
        Bias-corrected application data.
    """

    # Ensure all input arrays use the same dimension order
    Y_train = Y_train.transpose("time", "latitude", "longitude")

    X_train = X_train.transpose("time", "latitude", "longitude")

    X_apply = X_apply.transpose("time", "latitude", "longitude")

    # Get the number of latitude and longitude points
    n_lat = X_apply.sizes["latitude"]
    n_lon = X_apply.sizes["longitude"]

    # Allocate the full output array that will store corrected values
    Z_apply = np.empty(X_apply.shape, dtype=np.float32)

    # Compute the total number of chunks
    total_chunks = int(np.ceil(n_lat / chunk_lat)) * int(np.ceil(n_lon / chunk_lon))

    chunk_id = 0

    # Process each spatial chunk independently to reduce memory usage
    for lat_slice, lon_slice in iter_spatial_chunks(
        n_lat=n_lat, n_lon=n_lon, chunk_lat=chunk_lat, chunk_lon=chunk_lon
    ):
        chunk_id += 1

        logger.info(
            f"Processing spatial chunk "
            f"{chunk_id}/{total_chunks} | "
            f"lat={lat_slice}, lon={lon_slice}"
        )

        logger.info("Reading ERA5 training chunk")

        # Read the ERA5 reference data for the current spatial chunk
        Y_chunk = Y_train.isel(latitude=lat_slice, longitude=lon_slice).values.astype(
            np.float32
        )

        logger.info("Reading CMIP6 training chunk")

        # Read the CMIP6 training data for the same spatial chunk
        X_train_chunk = X_train.isel(
            latitude=lat_slice, longitude=lon_slice
        ).values.astype(np.float32)

        logger.info("Reading CMIP6 application chunk")

        # Read the CMIP6 data to be corrected for the same spatial chunk
        X_apply_chunk = X_apply.isel(
            latitude=lat_slice, longitude=lon_slice
        ).values.astype(np.float32)

        # Reshape data from 3D: time x latitude x longitude
        # to 2D: time x grid_points, as required by the QM model
        Y_train_2D = Y_chunk.reshape(Y_chunk.shape[0], -1)

        X_train_2D = X_train_chunk.reshape(X_train_chunk.shape[0], -1)

        X_apply_2D = X_apply_chunk.reshape(X_apply_chunk.shape[0], -1)

        logger.info(f"Y_train_2D shape: {Y_train_2D.shape}")

        logger.info(f"X_train_2D shape: {X_train_2D.shape}")

        logger.info(f"X_apply_2D shape: {X_apply_2D.shape}")

        # Create a new Quantile Mapping model for this spatial chunk
        qm = QM()

        # Fit Quantile Mapping using ERA5 as reference and CMIP6 as model dat
        qm.fit(Y0=Y_train_2D, X0=X_train_2D)

        # Apply the fitted Quantile Mapping model to the application data
        Z_chunk_2D = qm.predict(X0=X_apply_2D)

        # Reshape the corrected data back to the original 3D chunk shape
        Z_chunk = Z_chunk_2D.astype(np.float32).reshape(X_apply_chunk.shape)

        # Insert the corrected chunk into the full output array
        Z_apply[:, lat_slice, lon_slice] = Z_chunk

        # delete temporary arrays to reduce memory usage
        del Y_chunk
        del X_train_chunk
        del X_apply_chunk
        del Y_train_2D
        del X_train_2D
        del X_apply_2D
        del Z_chunk_2D
        del Z_chunk
        del qm

        # Force garbage collection after each chunk
        gc.collect()

    # Create a corrected xarray DataArray using the metadata of the input data
    corr = X_apply.copy(data=Z_apply)

    # Assign the variable name to the corrected DataArray
    corr.name = variable_name

    # Return the bias-corrected application data
    return corr


def main():
    """
    Main Quantile Mapping workflow.
    """

    args = parse_args()

    logger = Logger()

    if args.train_start > args.train_end:
        raise ValueError(
            f"train_start must be <= train_end, got "
            f"{args.train_start} > {args.train_end}"
        )

    if args.apply_start > args.apply_end:
        raise ValueError(
            f"apply_start must be <= apply_end, got "
            f"{args.apply_start} > {args.apply_end}"
        )

    train_years = args.train_end - args.train_start + 1

    logger.info(f"Opening ERA5 training data " f"({args.train_start}-{args.train_end})")

    # Load ERA5 reference files for the full training period
    era5_paths = build_era5_paths(
        start_year=args.train_start, end_year=args.train_end, era5_root=args.era5_root
    )

    era5_train_ds = xr.open_mfdataset(
        era5_paths, combine="nested", concat_dim="time", engine="netcdf4", cache=False
    )

    Y_train = era5_train_ds[args.variable]

    logger.info(
        f"Opening preprocessed CMIP6 training Zarr:\n" f"{args.cmip6_train_zarr}"
    )

    # Load preprocessed CMIP6 data used to train the correction
    cmip6_train_ds = xr.open_zarr(args.cmip6_train_zarr)

    X_train = cmip6_train_ds[args.variable]

    logger.info(
        f"Opening preprocessed CMIP6 application Zarr:\n" f"{args.cmip6_apply_zarr}"
    )

    # Load preprocessed CMIP6 data for the year that will be bias-corrected
    cmip6_apply_ds = xr.open_zarr(args.cmip6_apply_zarr)

    X_apply = cmip6_apply_ds[args.variable]

    X_apply = select_year_range(
        da=X_apply,
        start_year=args.apply_start,
        end_year=args.apply_end,
    )

    logger.info(f"ERA5 training shape       : {Y_train.shape}")

    logger.info(f"CMIP6 training shape      : {X_train.shape}")

    logger.info(f"CMIP6 application shape   : {X_apply.shape}")

    # ERA5 reference and CMIP6 training data must be aligned in time and space
    if Y_train.shape != X_train.shape:
        raise ValueError(
            "ERA5 training data and CMIP6 training data "
            f"do not have the same shape: "
            f"{Y_train.shape} != {X_train.shape}"
        )

    logger.info(
        f"Applying chunked Quantile Mapping "
        f"trained on {train_years} year(s) "
        f"({args.train_start}-{args.train_end}) "
        f"to application period "
        f"({args.apply_start}-{args.apply_end})"
    )

    # Apply Quantile Mapping correction to the application year
    corr = apply_qm_by_spatial_chunks(
        Y_train=Y_train,
        X_train=X_train,
        X_apply=X_apply,
        variable_name=args.variable,
        chunk_lat=args.chunk_lat,
        chunk_lon=args.chunk_lon,
        logger=logger,
    )

    # Save the corrected data as a NetCDF file
    save_corrected_by_year(
        corr=corr,
        output_dir=args.output_dir,
        start_year=args.apply_start,
        end_year=args.apply_end,
        logger=logger,
    )

    logger.success("Corrected datasets successfully saved")

    # Close opened datasets to release file handles and free resources.
    era5_train_ds.close()
    cmip6_train_ds.close()
    cmip6_apply_ds.close()


if __name__ == "__main__":
    main()
