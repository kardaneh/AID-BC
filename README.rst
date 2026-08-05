AID-BC
======

AI Downscaling and Bias Correction (AID-BC) is a Python package for
bias correction of climate-model data on the native CMIP6 grid.

The package provides:

- loading and standardization of yearly ERA5 and CMIP6 NetCDF files;
- resizing ERA5 fields onto the native CMIP6 grid;
- optional precomputation of ERA5-on-CMIP6 yearly files;
- univariate Quantile Mapping (QM);
- univariate or multivariate Optimal Transport (OT);
- global or calendar-month bias-correction fitting;
- yearly compressed NetCDF outputs;
- logging and diagnostic utilities.


Main concepts
-------------

Native CMIP6 grid
~~~~~~~~~~~~~~~~~

Bias correction is performed on the native CMIP6 spatial grid.

During training, ERA5 is used as the reference dataset. ERA5 can either be:

- loaded from its original grid and resized onto the CMIP6 grid during the
  bias-correction run; or
- precomputed once onto the CMIP6 grid with ``AID_BC.preprocess`` and reused
  in later runs.

CMIP6 data remain on their native grid throughout the workflow.


Bias-correction methods
~~~~~~~~~~~~~~~~~~~~~~~

Two methods are available:

``qm``
   Quantile Mapping. This method is univariate and therefore accepts exactly
   one climate variable.

``ot``
   Optimal Transport. This method accepts one or several climate variables.
   All requested variables and spatial grid points are concatenated into a
   common feature space and transported jointly.


Split modes
~~~~~~~~~~~

Two fitting strategies are available:

``none``
   Fit one corrector using all training samples.

``month``
   Fit one separate corrector for each calendar month. This can better preserve
   seasonality and substantially reduce the number of samples involved in each
   OT problem.


Input data layout
-----------------

Input directories contain one NetCDF file per year:

.. code-block:: text

   root/
   ├── samples_1980.nc
   ├── samples_1981.nc
   ├── ...
   └── samples_2014.nc

Each climate variable must contain the dimensions:

.. code-block:: text

   time, latitude, longitude

CMIP-style coordinate names ``lat`` and ``lon`` are accepted and standardized
internally to ``latitude`` and ``longitude``.

Longitudes are normalized to the interval ``[0, 360)`` and sorted. Latitude
ordering is standardized consistently between ERA5, historical CMIP6, and
application-period CMIP6 data.


Repository structure
--------------------

.. code-block:: text

   src/
   └── AID_BC/
       ├── __init__.py
       ├── __main__.py
       ├── bias_corrector.py       Common interface for QM and OT correctors.
       ├── dataset.py              ERA5 and CMIP6 loading and grid preparation.
       ├── diagnostics.py          Diagnostic tools for corrected datasets.
       ├── logger.py               Logging utilities.
       ├── main.py                 Main bias-correction command-line workflow.
       ├── optimal_transport.py    Sinkhorn-based Optimal Transport correction.
       ├── preprocess.py           ERA5 resizing onto the native CMIP6 grid.
       ├── quantile_mapping.py     Empirical and parametric Quantile Mapping.
       └── version.py              Package version information.

   tests/
   ├── test_dataset.py             Tests for dataset loading and preparation.
   ├── test_logger.py              Tests for logging utilities.
   ├── test_preprocess.py          Tests for ERA5-on-CMIP6 preprocessing.
   ├── test_quantile_mapping.py    Tests for Quantile Mapping components.
   └── test_runner.py              Entry point for running the complete test suite.


Workflow
--------

The workflow contains an optional preprocessing step followed by
bias correction.


1. Precompute ERA5 on the CMIP6 grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AID_BC.preprocess`` resizes one ERA5 variable onto the native grid defined by
a CMIP6 dataset and writes one compressed NetCDF file per year.

Run this step once for each combination of:

- ERA5 source directory;
- CMIP6 grid;
- climate variable.

Example for 2 m air temperature:

.. code-block:: bash

   python -m AID_BC.preprocess \
     --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
     --cmip6_root /data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
     --variable VAR_2T \
     --start_year 1980 \
     --end_year 2014 \
     --output_dir /data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_t2m

The output directory contains:

.. code-block:: text

   data_6hourly_t2m/
   ├── samples_1980.nc
   ├── samples_1981.nc
   ├── ...
   ├── samples_2014.nc
   └── metadata.json

The metadata file records:

- the variable name;
- the original ERA5 latitude orientation;
- the ERA5 and CMIP6 source directories;
- the processed year range.

For an OT experiment using several variables, run the preprocessing command
once per variable, for example:

.. code-block:: text

   /data/kkingston/data/CMIP6/ERA5_on_CMIP6/
   ├── data_6hourly_t2m/
   │   ├── metadata.json
   │   └── samples_YYYY.nc
   ├── data_6hourly_u10/
   │   ├── metadata.json
   │   └── samples_YYYY.nc
   └── data_6hourly_v10/
       ├── metadata.json
       └── samples_YYYY.nc


2. Run Quantile Mapping
~~~~~~~~~~~~~~~~~~~~~~~

QM accepts exactly one variable.

The command-line path arguments use the ``VARIABLE=PATH`` format.

Example using precomputed ERA5:

.. code-block:: bash

   python -m AID_BC.main \
     --method qm \
     --split month \
     --train_start 1980 \
     --train_end 2014 \
     --apply_start 2015 \
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


3. Run Optimal Transport
~~~~~~~~~~~~~~~~~~~~~~~~

OT can correct one variable or several variables jointly.

Example for 2 m temperature and the two 10 m wind components:

.. code-block:: bash

   python -m AID_BC.main \
     --method ot \
     --split month \
     --train_start 1980 \
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
     --ot_epsilon 1000 \
     --ot_num_iterations 15000 \
     --ot_threshold 0.01 \
     --ot_batch_size 16 \
     --ot_dtype float64

The OT parameters used in this example may need to be adjusted depending on
the variables, sample size, normalization, and available computational
resources.

By default, OT applies the same fitted per-feature normalization to the ERA5
reference data, the historical CMIP6 training data, and the CMIP6 application
data. Use ``--ot_no_normalization`` to disable it.


Output files
------------

Each corrected variable is written separately to the directory assigned through
``--output_dir``.

The output layout is:

.. code-block:: text

   output_directory/
   ├── samples_2015.nc
   ├── samples_2016.nc
   ├── ...
   └── samples_2021.nc

Files are written as compressed ``float32`` NetCDF data with dimensions:

.. code-block:: text

   time, latitude, longitude


Processing details
------------------

Training
~~~~~~~~

For every training year, AID-BC loads:

- ERA5 reference data, either raw or precomputed on the CMIP6 grid;
- historical CMIP6 biased data.

The package verifies that requested variables share:

- the same spatial grid;
- the same time coordinates within each dataset;
- finite values;
- the required dimensions.

The yearly datasets are concatenated along the time dimension before fitting.


Application
~~~~~~~~~~~

For every application year, AID-BC:

1. loads CMIP6 data on the native grid;
2. applies the fitted global or monthly corrector;
3. reconstructs each corrected variable on the original
   ``time × latitude × longitude`` layout;
4. writes one NetCDF file per corrected variable and year.


Quantile Mapping
~~~~~~~~~~~~~~~~

QM is fitted independently for each feature. In the current workflow, one
variable is flattened from:

.. code-block:: text

   time × latitude × longitude

to:

.. code-block:: text

   time × features

The fitted empirical mapping transforms biased CMIP6 quantiles into ERA5
reference quantiles.


Optimal Transport
~~~~~~~~~~~~~~~~~

For OT, all requested variables are flattened and concatenated:

.. code-block:: text

   time × (variables × latitude × longitude)

The resulting multivariate samples are transported jointly using an
entropy-regularized Sinkhorn solver.

The main OT options are:

``--ot_epsilon``
   Entropic regularization strength.

``--ot_num_iterations``
   Maximum number of Sinkhorn iterations.

``--ot_threshold``
   Convergence threshold.

``--ot_batch_size``
   Number of application time steps transformed together.

``--ot_dtype``
   JAX floating-point type: ``float32`` or ``float64``.

``--ot_no_normalization``
   Disable shared per-feature normalization.


Testing
-------

Run the complete test suite from the repository root:

.. code-block:: bash

   python -m tests.test_runner

Run an individual test module:

.. code-block:: bash

   python -m unittest tests.test_dataset
   python -m unittest tests.test_logger
   python -m unittest tests.test_preprocess
   python -m unittest tests.test_quantile_mapping
