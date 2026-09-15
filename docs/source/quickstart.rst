Quickstart
==========

Basic Workflow
--------------

1. **Setup environment** (see :doc:`installation`)
2. **Test components** (see :doc:`testing_philosophy`)
3. **Preprocess ERA5 data**
4. **Apply Quantile Mapping** (see :doc:`quantile_mapping`)
5. **Apply Optimal Transport** (see :doc:`optimal_transport`)

Command-Line Help
-----------------

Use the following commands to display the available options:

.. code-block:: bash

   python -m AID_BC.preprocess --help
   python -m AID_BC.main --help

Preprocess ERA5 Data
--------------------

Resize ERA5 data from its native grid (721 x 1440) onto the native CMIP6 grid
(143 x 144) before applying bias correction. The preprocessing step writes one
NetCDF file per year and should be run once for each climate variable:

.. code-block:: bash

   python -m AID_BC.preprocess \
      --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
      --cmip6_root /data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
      --variable VAR_2T \
      --start_year 1980 \
      --end_year 2014 \
      --output_dir /data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_t2m

The preprocessed ERA5 data can then be reused for Quantile Mapping or Optimal
Transport runs.

Apply Quantile Mapping
----------------------

Apply Quantile Mapping using preprocessed ERA5 data as the reference dataset.
Quantile Mapping accepts exactly one climate variable:

.. code-block:: bash

   python -m AID_BC.main \
      --method qm \
      --split month \
      --train_start 1980 \
      --train_end 2014 \
      --apply_start 2021 \
      --apply_end 2021 \
      --variables VAR_2T \
      --era5_on_cmip6_root \
        VAR_2T=/data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_t2m \
      --cmip6_train_root \
        VAR_2T=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
      --cmip6_apply_root \
        VAR_2T=/data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_tas \
      --output_dir \
        VAR_2T=/net/nfs/ssd1/kkingston/AID-BC/data/CMIP6_QM/data_6hourly_tas_corrected

Apply Optimal Transport
-----------------------

Apply Optimal Transport using preprocessed ERA5 data as the reference dataset.
Optimal Transport can correct one or several climate variables jointly:

.. code-block:: bash

   python -m AID_BC.main \
      --method ot \
      --split month \
      --train_start 2000 \
      --train_end 2014 \
      --apply_start 2021 \
      --apply_end 2021 \
      --variables VAR_2T VAR_10U VAR_10V \
      --era5_on_cmip6_root \
        VAR_2T=/data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_t2m \
        VAR_10U=/data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_u10 \
        VAR_10V=/data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_v10 \
      --cmip6_train_root \
        VAR_2T=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
        VAR_10U=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_uas \
        VAR_10V=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_vas \
      --cmip6_apply_root \
        VAR_2T=/data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_tas \
        VAR_10U=/data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_uas \
        VAR_10V=/data/kkingston/data/CMIP6/CMIP6_futur/data_6hourly_vas \
      --output_dir \
        VAR_2T=/net/nfs/ssd1/kkingston/AID-BC/data/CMIP6_OT/data_6hourly_tas_corrected \
        VAR_10U=/net/nfs/ssd1/kkingston/AID-BC/data/CMIP6_OT/data_6hourly_uas_corrected \
        VAR_10V=/net/nfs/ssd1/kkingston/AID-BC/data/CMIP6_OT/data_6hourly_vas_corrected \
      --ot_epsilon 1 \
      --ot_num_iterations 1000000 \
      --ot_threshold 0.1 \
      --ot_batch_size 16 \
      --ot_dtype float64

The corrected CMIP6 data are saved as yearly compressed NetCDF files in the
output directories specified with ``--output_dir``.
