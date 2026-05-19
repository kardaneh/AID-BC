# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
from pathlib import Path

import numpy as np
import gc

from dataset import ClimateDataset
from logger import Logger
from quantile_mapping import QM


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Quantile Mapping bias correction")

    # First year used to train the QM correction
    parser.add_argument(
        "--train_start", type=int, required=True, help="Training start year"
    )

    # Last year used to train the QM correction
    parser.add_argument(
        "--train_end", type=int, required=True, help="Training end year"
    )

    parser.add_argument(
        "--apply_year", type=int, required=True, help="Application year"
    )

    parser.add_argument("--variable", type=str, default="VAR_2T", help="Variable name")

    parser.add_argument(
        "--era5_root", type=str, required=True, help="ERA5 root directory"
    )

    parser.add_argument(
        "--cmip6_train_root",
        type=str,
        required=True,
        help="CMIP6 historical root directory",
    )

    parser.add_argument(
        "--cmip6_apply_root",
        type=str,
        required=True,
        help="CMIP6 future root directory",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    # Number of latitude points processed at once
    parser.add_argument("--chunk_lat", type=int, default=64, help="Latitude chunk size")

    # Number of longitude points processed at once
    parser.add_argument(
        "--chunk_lon", type=int, default=64, help="Longitude chunk size"
    )

    return parser.parse_args()


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

            # Yield the latitude and longitude slices for this chunk
            yield (slice(lat_start, lat_end), slice(lon_start, lon_end))


def apply_qm_by_spatial_chunks(
    train_datasets, ds_apply, variable_name, chunk_lat, chunk_lon, logger
):
    """
    Apply Quantile Mapping chunk by chunk.

    Parameters
    ----------
    train_datasets : list[ClimateDataset]
        Prepared training datasets.

    ds_apply : ClimateDataset
        Prepared application dataset.

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

    # Reorder the application data dimensions to ensure a consistent layout
    apply_data = ds_apply.cmip6_data.transpose("time", "latitude", "longitude")

    # Get the number of latitude and longitude points
    n_lat = apply_data.sizes["latitude"]
    n_lon = apply_data.sizes["longitude"]

    # Allocate the full output array that will store corrected values
    Z_apply = np.empty(apply_data.shape, dtype=np.float32)

    # Compute the total number of chunks
    total_chunks = int(np.ceil(n_lat / chunk_lat)) * int(np.ceil(n_lon / chunk_lon))

    chunk_id = 0

    # Iterate over all spatial chunks of the grid
    for lat_slice, lon_slice in iter_spatial_chunks(
        n_lat=n_lat, n_lon=n_lon, chunk_lat=chunk_lat, chunk_lon=chunk_lon
    ):
        chunk_id += 1

        logger.info(
            f"Processing spatial chunk "
            f"{chunk_id}/{total_chunks} | "
            f"lat={lat_slice}, lon={lon_slice}"
        )

        # Store ERA5 and CMIP6 training chunks for all training years
        Y_train_chunks = []
        X_train_chunks = []

        # Read the corresponding spatial chunk from each training dataset
        for ds_train in train_datasets:
            logger.info(f"Reading ERA5 chunk from {ds_train.era5_path}")

            Y_chunk = (
                ds_train.era5_data.transpose("time", "latitude", "longitude")
                .isel(latitude=lat_slice, longitude=lon_slice)
                .values
            )

            logger.info(f"Reading CMIP6 chunk from {ds_train.cmip6_path}")

            X_chunk = (
                ds_train.cmip6_data.transpose("time", "latitude", "longitude")
                .isel(latitude=lat_slice, longitude=lon_slice)
                .values
            )

            # Add the ERA5 chunk to the list of observed training chunks
            Y_train_chunks.append(Y_chunk)

            # Add the CMIP6 chunk to the list of model training chunks
            X_train_chunks.append(X_chunk)

        # Concatenate
        Y_train = np.concatenate(Y_train_chunks, axis=0)

        X_train = np.concatenate(X_train_chunks, axis=0)

        logger.info(f"Reading apply CMIP6 chunk from {ds_apply.cmip6_path}")

        # Extract the CMIP6 chunk to be corrected
        X_apply = apply_data.isel(latitude=lat_slice, longitude=lon_slice).values

        # Flatten the spatial dimensions of the ERA5 training data
        # Shape becomes: time x spatial_points
        Y_train_2D = Y_train.reshape(Y_train.shape[0], -1)

        # Flatten the spatial dimensions of the CMIP6 training data
        X_train_2D = X_train.reshape(X_train.shape[0], -1)

        # Flatten the spatial dimensions of the CMIP6 application data
        X_apply_2D = X_apply.reshape(X_apply.shape[0], -1)

        # Create a new Quantile Mapping model for this spatial chunk
        qm = QM()

        # Fit Quantile Mapping using ERA5 as reference and CMIP6 as model data
        qm.fit(Y0=Y_train_2D, X0=X_train_2D)

        # Apply the fitted Quantile Mapping model to the application data
        Z_chunk_2D = qm.predict(X0=X_apply_2D)

        # Convert corrected data to float32 and restore the original chunk shape
        Z_chunk = Z_chunk_2D.astype(np.float32).reshape(X_apply.shape)

        # Insert the corrected chunk into the full output array
        Z_apply[:, lat_slice, lon_slice] = Z_chunk

        # delete temporary arrays to reduce memory usage
        del Y_train_chunks
        del X_train_chunks
        del Y_train
        del X_train
        del X_apply
        del Y_train_2D
        del X_train_2D
        del X_apply_2D
        del Z_chunk_2D
        del Z_chunk
        del qm

        # Force garbage collection after each chunk
        gc.collect()

    # Create a corrected xarray DataArray using the metadata of the input data
    corr = apply_data.copy(data=Z_apply)

    # Assign the variable name to the corrected DataArray
    corr.name = variable_name

    # Return the bias-corrected application data
    return corr


def build_paths(year, era5_root, cmip6_root):
    """
    Build ERA5 and CMIP6 file paths.

    Parameters
    ----------
    year : int
        Dataset year.

    era5_root : str
        ERA5 root directory.

    cmip6_root : str
        CMIP6 root directory.

    Returns
    -------
    tuple[str, str]
        ERA5 and CMIP6 file paths.
    """

    era5_path = Path(era5_root) / f"samples_{year}.nc"

    cmip6_path = Path(cmip6_root) / f"samples_{year}.nc"

    return str(era5_path), str(cmip6_path)


def load_dataset(year, era5_root, cmip6_root, variable_name, logger):
    """
    Load and prepare climate dataset.

    Parameters
    ----------
    year : int
        Dataset year.

    era5_root : str
        ERA5 root directory.

    cmip6_root : str
        CMIP6 root directory.

    variable_name : str
        Variable name.

    logger : Logger
        Logger instance.

    Returns
    -------
    ClimateDataset
        Prepared dataset.
    """

    # Build the ERA5 and CMIP6 paths for the selected year
    era5_path, cmip6_path = build_paths(
        year=year, era5_root=era5_root, cmip6_root=cmip6_root
    )

    # Create a ClimateDataset instance for this year
    ds = ClimateDataset(
        era5_path=era5_path,
        cmip6_path=cmip6_path,
        variable_name=variable_name,
        logger=logger,
    )

    # Run the complete dataset preparation pipeline
    ds.prepare()

    return ds


def main():
    """
    Main Quantile Mapping workflow.
    """

    args = parse_args()

    logger = Logger()

    logger.info("Loading training datasets")

    train_datasets = []

    # Loop over all years included in the training period
    for year in range(args.train_start, args.train_end + 1):
        logger.info(f"Loading training year {year}")

        # Load and prepare the dataset for the current training year
        ds_train = load_dataset(
            year=year,
            era5_root=args.era5_root,
            cmip6_root=args.cmip6_train_root,
            variable_name=args.variable,
            logger=logger,
        )

        train_datasets.append(ds_train)

    train_years = args.train_end - args.train_start + 1

    logger.info(
        f"Loaded {train_years} training year(s) "
        f"({args.train_start}-{args.train_end})"
    )

    logger.info(f"Loading application year {args.apply_year}")

    ds_apply = load_dataset(
        year=args.apply_year,
        era5_root=args.era5_root,
        cmip6_root=args.cmip6_apply_root,
        variable_name=args.variable,
        logger=logger,
    )

    logger.info(
        f"Applying chunked Quantile Mapping "
        f"trained on {train_years} year(s) "
        f"({args.train_start}-{args.train_end})"
    )

    # Apply Quantile Mapping correction chunk by chunk
    corr = apply_qm_by_spatial_chunks(
        train_datasets=train_datasets,
        ds_apply=ds_apply,
        variable_name=args.variable,
        chunk_lat=args.chunk_lat,
        chunk_lon=args.chunk_lon,
        logger=logger,
    )

    # Build the output NetCDF file path
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"samples_{args.apply_year}.nc"

    logger.info(f"Saving corrected dataset to:\n{output_file}")

    # Save the corrected data as a NetCDF file
    corr.to_netcdf(output_file)

    logger.success("Corrected dataset successfully saved")


if __name__ == "__main__":
    main()
