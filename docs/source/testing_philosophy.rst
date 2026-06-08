Testing Philosophy
===================

AID-BC uses Python's built-in `unittest` framework for comprehensive testing.

Before running bias-correction workflows on large ERA5 and CMIP6 datasets,
users should first validate the main components with the test suite.

Why Testing is Essential
------------------------

AID-BC workflows involve data loading, coordinate handling, interpolation,
statistical correction, and large multidimensional datasets.

Testing helps prevent:

- invalid scientific results,
- data-processing errors,
- wasted computational resources,
- reproducibility issues.

Recommended Workflow
--------------------

1. Run the complete test suite:

   .. code-block:: bash

      python -m tests.test_runner

2. Run a specific test module:

   .. code-block:: bash

      python -m unittest tests.test_dataset

3. Run a specific test class:

   .. code-block:: bash

      python -m unittest tests.test_dataset.TestClimateDataset

4. Run a single test method:

   .. code-block:: bash

      python -m unittest \
          tests.test_dataset.TestClimateDataset.test_prepare_pipeline


Test Coverage
-------------

The AID-BC tests cover:

- dataset initialization and loading,
- missing-variable errors,
- coordinate renaming and longitude handling,
- CMIP6 interpolation onto the ERA5 grid,
- yearly preprocessing,
- path construction,
- single-year and multi-year Zarr outputs,
- overwrite protection,
- complete preparation pipelines.

Tests use small temporary NetCDF datasets so they remain fast, isolated, and
independent from production data.
