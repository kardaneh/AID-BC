Overview
========

AID-BC (AI Downscaling and Bias Correction) is a Python framework for
preparing and bias-correcting climate data before AI-based downscaling.

The package is designed for workflows in which climate model outputs must be
made more consistent with a reference dataset before being used as inputs to an
AI downscaling model. ERA5 is used as the reference dataset, while historical
CMIP6 data are used to fit the bias-correction model and future or independent
CMIP6 data are used during application.

Bias correction is performed on the native CMIP6 latitude-longitude grid. ERA5
fields are either processed directly onto the CMIP6 grid during the workflow or
loaded from previously prepared ERA5-on-CMIP6 files.

AID-BC currently supports two bias-correction approaches:

- univariate Quantile Mapping,
- multivariate entropy-regularized Optimal Transport.

Quantile Mapping corrects one climate variable independently. Optimal Transport
can correct one or several variables jointly by representing all selected
variables and spatial grid points in a common multivariate feature space.

The corrected CMIP6 fields are saved as yearly NetCDF files and can then be used
as physically more consistent large-scale inputs for downstream AI downscaling
experiments.

Main objectives
---------------

AID-BC provides tools to:

- load ERA5 and CMIP6 climate datasets,
- process ERA5 data onto the native CMIP6 grid,
- load previously prepared ERA5-on-CMIP6 data,
- validate spatial grids, temporal coordinates, and variable consistency,
- apply univariate Quantile Mapping,
- apply multivariate entropy-regularized Optimal Transport,
- jointly correct several climate variables with Optimal Transport,
- fit one global corrector or one corrector per month,
- process application data one year at a time,
- save each corrected variable as a compressed yearly NetCDF file,
- prepare corrected CMIP6 inputs for AI-based downscaling workflows.

Supported methods
-----------------

1. Quantile Mapping
~~~~~~~~~~~~~~~~~~~

Quantile Mapping is implemented as a univariate correction method. It requires
exactly one climate variable for each execution.

The method learns the statistical relationship between historical CMIP6 data
and ERA5 reference data over the selected training period. The fitted mapping
is then applied to CMIP6 data from the requested application period.

Quantile Mapping corrects the marginal distribution of the selected variable
but does not explicitly model dependencies between several variables.

2. Optimal Transport
~~~~~~~~~~~~~~~~~~~~

Optimal Transport is implemented as a multivariate bias-correction method. It
can be applied to one variable or to several variables jointly.

The selected variables are flattened over the spatial grid and concatenated
into a common feature matrix. Each time step therefore becomes one
multidimensional sample containing all selected variables and all native CMIP6
grid points.

An entropy-regularized Sinkhorn solver is fitted between:

- historical CMIP6 samples as the source distribution,
- ERA5 samples on the CMIP6 grid as the target distribution.

The fitted transport map is then applied to the CMIP6 application period.

This formulation allows the correction to account for dependencies between
variables and spatial grid points within the multivariate feature space.

Native CMIP6 grid
-----------------

AID-BC performs bias correction on the native CMIP6 grid.

When raw ERA5 files are used, ERA5 data are prepared and aligned with the CMIP6
grid before fitting. When ERA5-on-CMIP6 files have already been generated, the
workflow loads these precomputed fields directly.

All selected variables must share compatible:

- time coordinates,
- latitude coordinates,
- longitude coordinates,
- spatial dimensions.

The same variable ordering is preserved during training, application,
flattening, transport, reconstruction, and output.

Workflow summary
----------------

The workflow is divided into four main stages.

1. Argument and path validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command-line configuration defines:

- the bias-correction method,
- the temporal split mode,
- the training period,
- the application period,
- the selected climate variables,
- the ERA5 input directories,
- the historical CMIP6 directories,
- the CMIP6 application directories,
- the output directories,
- the Optimal Transport parameters when applicable.

Variable-specific paths are provided using the following format:

.. code-block:: text

   VARIABLE=PATH

For example:

.. code-block:: text

   VAR_2T=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas
   VAR_10U=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_uas
   VAR_10V=/data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_vas

The workflow verifies that paths are provided for exactly the requested
variables.

2. Training-data preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each selected variable, AID-BC loads:

- ERA5 reference data,
- historical CMIP6 training data.

ERA5 can be provided either as:

- raw ERA5 data that must be prepared on the CMIP6 grid,
- precomputed ERA5-on-CMIP6 yearly files.

The yearly fields are loaded, checked, and concatenated across the complete
training period.

When several variables are requested, they are aligned exactly in time,
latitude, and longitude before being assembled into common xarray datasets.

3. Bias-corrector fitting
~~~~~~~~~~~~~~~~~~~~~~~~~

The training datasets are flattened into matrices with shape:

.. code-block:: text

   (number_of_time_steps, number_of_features)

For one variable, the features correspond to all latitude-longitude grid
points.

For multivariate Optimal Transport, the feature blocks of all requested
variables are concatenated in the order specified on the command line.

The workflow can fit:

- one global corrector using all training samples,
- one separate corrector for each month.

Monthly splitting reduces the number of samples included in each fit and can
substantially reduce the computational cost of Optimal Transport.

4. Application and output
~~~~~~~~~~~~~~~~~~~~~~~~~

CMIP6 application data are loaded one year at a time.

The fitted global or monthly corrector is applied to the corresponding
application samples. The corrected feature matrix is then reconstructed into
the original time, latitude, and longitude dimensions.

Each corrected variable is written separately as:

.. code-block:: text

   samples_<year>.nc

The output files use NetCDF compression and preserve the variable coordinates
and relevant metadata from the CMIP6 application data.

Temporal splitting
------------------

AID-BC supports two temporal fitting modes.

Global correction
~~~~~~~~~~~~~~~~~

With:

.. code-block:: text

   --split none

one corrector is fitted using all training samples across all months.

Monthly correction
~~~~~~~~~~~~~~~~~~

With:

.. code-block:: text

   --split month

one independent corrector is fitted for each month.

During application, January data are corrected with the January corrector,
February data with the February corrector, and so on. The corrected monthly
datasets are then concatenated and restored to chronological order.

Monthly splitting is particularly useful for climate variables with strong
seasonal distributions. It also reduces the number of samples involved in each
Optimal Transport problem.
