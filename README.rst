AID-BC
======

AI Downscaling and Bias Correction (AID-BC) is a Python package for bias
correction of climate data using machine learning techniques.

The package currently provides tools for:

- loading and handling climate datasets;
- preprocessing CMIP6 data onto the ERA5 grid;
- saving preprocessed CMIP6 data as Zarr files;
- applying Quantile Mapping for bias correction;
- running chunked spatial bias-correction workflows;
- testing dataset utilities.

Repository structure
--------------------

.. code-block:: text

   src/
   └── AID_BC/
       ├── __init__.py
       ├── __main__.py
       ├── dataset.py
       ├── logger.py
       ├── main.py
       ├── preprocess.py
       ├── quantile_mapping.py
       └── version.py

   notebooks/
   └── BSCK_2013_2014.ipynb

   tests/
   ├── test_dataset.py
   ├── test_logger.py
   └── test_runner.py


Workflow
--------

The bias-correction workflow is split into two main steps.

1. Preprocess CMIP6 data
~~~~~~~~~~~~~~~~~~~~~~~~

CMIP6 data are first interpolated onto the ERA5 grid and saved as Zarr files.
This avoids repeating the interpolation step during Quantile Mapping.

Example for the historical training period:

.. code-block:: bash

   python -m AID_BC.preprocess \
     --start_year 2000 \
     --end_year 2014 \
     --variable VAR_2T \
     --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
     --cmip6_root /data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
     --output_zarr /net/nfs/ssd1/kkingston/backup/AID-BC/data/zarr/cmip6_train_2000_2014.zarr \
     --time_chunk 1464 \
     --lat_chunk 144 \
     --lon_chunk 360 \
     --overwrite

Example for the application year:

.. code-block:: bash

   python -m AID_BC.preprocess \
     --start_year 2021 \
     --end_year 2021 \
     --variable VAR_2T \
     --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
     --cmip6_root /data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_tas \
     --output_zarr /net/nfs/ssd1/kkingston/backup/AID-BC/data/zarr/cmip6_apply_2021.zarr \
     --time_chunk 1464 \
     --lat_chunk 144 \
     --lon_chunk 360 \
     --overwrite

The preprocessing step performs:

- loading ERA5 and CMIP6 NetCDF files;
- renaming CMIP6 coordinates when needed;
- adding a cyclic longitude point;
- interpolating CMIP6 onto the ERA5 latitude-longitude grid;
- saving the interpolated CMIP6 data as Zarr.

2. Apply Quantile Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the CMIP6 training and application datasets have been preprocessed,
Quantile Mapping can be applied by spatial chunks.

.. code-block:: bash

   python -m AID_BC.main \
     --train_start 2000 \
     --train_end 2014 \
     --apply_year 2021 \
     --variable VAR_2T \
     --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
     --cmip6_train_zarr /net/nfs/ssd1/kkingston/backup/AID-BC/data/zarr/cmip6_train_2000_2014.zarr \
     --cmip6_apply_zarr /net/nfs/ssd1/kkingston/backup/AID-BC/data/zarr/cmip6_apply_2021.zarr \
     --output_dir /net/nfs/ssd1/kkingston/backup/AID-BC/data/data_6hourly_tas_corrected \
     --chunk_lat 144 \
     --chunk_lon 360

The correction step performs:

- lazy loading of ERA5 training data;
- loading of preprocessed CMIP6 Zarr data;
- spatial chunking over latitude and longitude;
- Quantile Mapping fitting on the training period;
- Quantile Mapping application to the target year;
- saving the corrected application year as NetCDF.


Chunking
--------

Chunking is used for two different purposes.

In ``preprocess.py``, the chunk arguments define how the preprocessed CMIP6 data
are stored on disk in Zarr format:

.. code-block:: bash

   --time_chunk 1464
   --lat_chunk 144
   --lon_chunk 360

This does not change the data values. It only controls the Zarr storage layout.

In ``main.py``, the chunk arguments define the spatial blocks used during
Quantile Mapping:

.. code-block:: bash

   --chunk_lat 144
   --chunk_lon 360

For each spatial block, all available time steps are used for training and
application. The spatial chunking limits memory usage during Quantile Mapping.


Notes
-----

The preprocessing step can significantly increase the size of CMIP6 data because
CMIP6 is interpolated from its native grid onto the higher-resolution ERA5 grid.

For example, a native CMIP6 field with shape similar to:

.. code-block:: text

   time x 143 x 144

is interpolated onto the ERA5 grid:

.. code-block:: text

   time x 721 x 1440

Therefore, the preprocessed Zarr files can be much larger than the original
CMIP6 NetCDF files.
