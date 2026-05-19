AID-BC
=========================================
AI Downscaling and Bias Correction (AID-BC) is a Python package for bias correction of climate data using machine learning techniques.

The package currently provides tools for:

- loading and handling climate datasets;
- applying quantile mapping for bias correction;
- running univariate bias-correction examples;
- testing dataset utilities.


Repository structure
--------------------

.. code-block:: text

   AID_BC/
   ├── dataset.py
   ├── logger.py
   ├── main.py
   ├── quantile_mapping.py
   └── version.py

   notebooks/
   └── BSCK_2013_2014.ipynb

   tests/
   └── test_dataset.py
   └── test_logger.py
   └── test_runner.py
