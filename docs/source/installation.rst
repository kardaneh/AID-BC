Installation
============

AID-BC uses **uv** for fast, portable, and reproducible environment management.

Prerequisites
-------------

- Python 3.8 or higher
- Git (for cloning the repository)

Quick Install
-------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/kardaneh/AID-BC.git
   cd AID-BC

2. Create and activate a virtual environment:

.. code-block:: bash

   uv venv --python=python3.11
   source .venv/bin/activate

3. Install the package in development mode:

.. code-block:: bash

   uv pip install -e .

This will install all dependencies defined in ``pyproject.toml`` and make the
``AID_BC`` command available in your environment.

Verification
------------

After installation, verify that the package is correctly installed:

.. code-block:: bash

   # Check help
   AID_BC --help

Optional Dependencies
---------------------

For development and documentation:

.. code-block:: bash

   # Install development dependencies
   uv pip install -e ".[dev]"

   # Install documentation dependencies
   uv pip install -e ".[docs]"

Development Installation
------------------------

For developers working on the codebase:

1. Install in editable mode with development dependencies:

.. code-block:: bash

   uv pip install -e ".[dev]"

2. Set up pre-commit hooks:

.. code-block:: bash

   pre-commit install
   pre-commit run --all-files

3. Run tests:

.. code-block:: bash

   python tests/test_runner.py

Getting Help
------------

- **Repository**: https://github.com/kardaneh/AID-BC
- **Issues**: https://github.com/kardaneh/AID-BC/issues
- **Contact**: kardaneh@ipsl.fr
