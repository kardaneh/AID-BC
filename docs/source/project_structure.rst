Project Structure
=================

The AID-BC package is organized as follows:

.. code-block:: text

   AID-BC/
   ├── src/AID_BC/                    # Main Python package
   │   ├── __init__.py
   │   ├── __main__.py
   │   ├── bias_corrector.py          # Common interface for QM and OT correctors
   │   ├── dataset.py                 # ERA5 and CMIP6 data loading and preprocessing
   │   ├── diagnostics.py             # Bias-correction diagnostic utilities
   │   ├── logger.py                  # Console and file logging
   │   ├── main.py                    # Main bias-correction workflow
   │   ├── optimal_transport.py       # Entropy-regularized Sinkhorn solver
   │   ├── preprocess.py              # ERA5 and CMIP6 preprocessing utilities
   │   ├── quantile_mapping.py        # Quantile Mapping implementation
   │   └── version.py                 # Package version information
   ├── tests/                         # Unit tests
   │   ├── test_dataset.py
   │   ├── test_logger.py
   │   ├── test_preprocess.py
   │   ├── test_quantile_mapping.py
   │   └── test_runner.py
   ├── docs/                          # Sphinx documentation
   │   ├── Makefile                   # Documentation build commands
   │   ├── requirements.txt           # Documentation dependencies
   │   └── source/
   │       ├── api/                   # API reference pages
   │       ├── conf.py                # Sphinx configuration
   │       ├── experiments.rst        # Experiment documentation
   │       ├── index.rst              # Documentation entry point
   │       ├── installation.rst       # Installation instructions
   │       ├── overview.rst           # Project overview
   │       ├── pre_push_workflow.rst  # Pre-push validation workflow
   │       ├── project_structure.rst  # Repository structure
   │       ├── quantile_mapping.rst   # Quantile Mapping documentation
   │       ├── quickstart.rst          # Quick-start guide
   │       └── testing_philosophy.rst # Testing strategy and philosophy
   ├── slurm/                         # Generated SLURM submission scripts
   ├── slurm_io/                      # SLURM standard output and error logs
   ├── data/                          # Generated bias-corrected datasets
   ├── setup                          # Bias-correction job configuration generator
   ├── pyproject.toml                 # Package and development configuration
   ├── README.rst                     # Main project documentation
   └── LICENSE                        # CC BY-NC-SA 4.0 license
