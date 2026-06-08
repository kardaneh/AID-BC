Quickstart
==========

Basic Workflow
--------------

1. **Setup environment** (see :doc:`installation`)
2. **Test components** (see :doc:`testing_philosophy`)
3. **Preprocess CMIP6 data**
4. **Apply Quantile Mapping**

Command-Line Help
-----------------

Use the following commands to display the available options:

.. code-block:: bash

   AID_BC --help

Preprocess CMIP6 Data
---------------------

Interpolate the CMIP6 data onto the ERA5 grid and save the result as a Zarr
dataset:

.. code-block:: bash

   python -m AID_BC.preprocess \
      --start_year 2022 \
      --end_year 2022 \
      --variable VAR_10V \
      --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_v10 \
      --cmip6_root /data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_vas \
      --output_zarr /scratchu/kkingston/AID-BC/data/zarr/cmip6_apply_2022_vas.zarr \
      --time_chunk 1464 \
      --lat_chunk 144 \
      --lon_chunk 360 \
      --overwrite

Apply Quantile Mapping
----------------------

Apply the bias correction using ERA5 as the reference dataset and the
preprocessed CMIP6 Zarr datasets:

.. code-block:: bash

   AID_BC \
      --train_start 1980 \
      --train_end 2014 \
      --apply_start 2022 \
      --apply_end 2022 \
      --variable VAR_10V \
      --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_v10 \
      --cmip6_train_zarr /scratchu/kkingston/AID-BC/data/zarr/cmip6_train_test_1980_2014_vas.zarr \
      --cmip6_apply_zarr /scratchu/kkingston/AID-BC/data/zarr/cmip6_apply_2022_vas.zarr \
      --output_dir /scratchu/kkingston/AID-BC/data/data_6hourly_vas_corrected \
      --chunk_lat 144 \
      --chunk_lon 360

The corrected CMIP6 data are saved as NetCDF files in the directory specified
by ``--output_dir``.
