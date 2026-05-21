# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import gc
import shutil
from pathlib import Path

from AID_BC.dataset import ClimateDataset
from AID_BC.logger import Logger


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Preprocess CMIP6 data onto ERA5 grid and save as Zarr"
    )

    parser.add_argument("--start_year", type=int, required=True, help="Start year")

    parser.add_argument("--end_year", type=int, required=True, help="End year")

    parser.add_argument("--variable", type=str, default="VAR_2T", help="Variable name")

    parser.add_argument(
        "--era5_root", type=str, required=True, help="ERA5 root directory"
    )

    parser.add_argument(
        "--cmip6_root", type=str, required=True, help="CMIP6 root directory"
    )

    parser.add_argument(
        "--output_zarr", type=str, required=True, help="Output Zarr path"
    )

    # Define the Zarr chunk sizes used for storage and later access
    parser.add_argument("--time_chunk", type=int, default=1460, help="Time chunk size")

    parser.add_argument(
        "--lat_chunk", type=int, default=144, help="Latitude chunk size"
    )

    parser.add_argument(
        "--lon_chunk", type=int, default=360, help="Longitude chunk size"
    )

    # Allow the output Zarr store to be replaced if it already exists
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output Zarr if it already exists",
    )

    return parser.parse_args()


def build_paths(year, era5_root, cmip6_root):
    """
    Build ERA5 and CMIP6 file paths.

    Parameters
    ----------
    year : int
        Year to process.

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


def preprocess_year(year, era5_root, cmip6_root, variable_name, logger):
    """
    Preprocess one CMIP6 year onto the ERA5 grid.

    Parameters
    ----------
    year : int
        Year to process.

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
    da : xarray.DataArray
        CMIP6 data interpolated onto ERA5 grid.
    """

    # Build the yearly input paths for ERA5 and CMIP6
    era5_path, cmip6_path = build_paths(
        year=year, era5_root=era5_root, cmip6_root=cmip6_root
    )

    # ClimateDataset handles loading, checking, and preparing ERA5/CMIP6 data
    ds = ClimateDataset(
        era5_path=era5_path,
        cmip6_path=cmip6_path,
        variable_name=variable_name,
        logger=logger,
    )

    # Prepare the data, including interpolation of CMIP6 onto the ERA5 grid
    ds.prepare()

    # Keep only the processed CMIP6 field and reduce precision to save memory
    da = ds.cmip6_data.astype("float32")

    da.name = variable_name

    # Close source NetCDF files after building the interpolated DataArray
    if ds.era5 is not None:
        ds.era5.close()

    if ds.cmip6 is not None:
        ds.cmip6.close()

    return da


def main():
    """
    Main preprocessing workflow.
    """

    args = parse_args()

    logger = Logger()

    output_zarr = Path(args.output_zarr)

    # Remove the existing Zarr store only if overwrite mode is enabled
    if output_zarr.exists():
        if args.overwrite:
            logger.info(f"Removing existing Zarr:\n{output_zarr}")

            shutil.rmtree(output_zarr)

        else:
            raise FileExistsError(
                f"Output Zarr already exists: {output_zarr}. "
                f"Use --overwrite to replace it."
            )

    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    first_year = True

    # Process each year independently to avoid loading the full period at once
    for year in range(args.start_year, args.end_year + 1):
        logger.info(f"Preprocessing CMIP6 year {year}")

        da = preprocess_year(
            year=year,
            era5_root=args.era5_root,
            cmip6_root=args.cmip6_root,
            variable_name=args.variable,
            logger=logger,
        )

        # Rechunk the data before writing to Zarr for storage and later processing
        da = da.chunk(
            {
                "time": args.time_chunk,
                "latitude": args.lat_chunk,
                "longitude": args.lon_chunk,
            }
        )

        # Convert the DataArray to a Dataset before saving it as Zarr
        ds_out = da.to_dataset(name=args.variable)

        logger.info(f"Writing year {year} to Zarr:\n{output_zarr}")

        if first_year:
            # Create the Zarr store for the first processed year
            ds_out.to_zarr(output_zarr, mode="w")

            first_year = False

        else:
            # Append following years along the time dimension
            ds_out.to_zarr(output_zarr, mode="a", append_dim="time")

        # Explicitly free memory after each year
        del da
        del ds_out
        # Force garbage collection
        gc.collect()

    logger.success("CMIP6 Zarr preprocessing completed")


if __name__ == "__main__":
    main()
