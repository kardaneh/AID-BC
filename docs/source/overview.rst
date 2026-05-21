Overview
========

AID-BC (AI Downscaling and Bias Correction) is a Python-based framework for
preparing and bias-correcting climate data before AI-based downscaling.

The package is designed to support workflows where climate model outputs must be
made consistent with a reference dataset before being used as inputs to a
downscaling model. In the current workflow, ERA5 is used as the reference
dataset and CMIP6 data are bias-corrected using Quantile Mapping.

The corrected CMIP6 fields can then be used as physically consistent large-scale
inputs for downstream AI downscaling experiments.

The current implementation focuses on univariate bias correction. Future
developments aim to extend the framework toward multivariate bias correction,
including methods based on Optimal Transport, in order to better preserve
dependencies between variables and spatial structures.

Main objectives
---------------

AID-BC provides tools to:

- load ERA5 and CMIP6 climate datasets,
- preprocess CMIP6 data onto the ERA5 latitude-longitude grid,
- store preprocessed CMIP6 data as Zarr files,
- apply Quantile Mapping by spatial chunks,
- reduce memory usage during large-scale bias correction,
- save corrected climate fields as NetCDF files,
- prepare corrected CMIP6 inputs for AI-based downscaling workflows,
- provide a foundation for future multivariate bias-correction methods,
- support future Optimal Transport-based bias-correction experiments.

Workflow summary
----------------

The workflow is split into two main steps.

1. Preprocessing
~~~~~~~~~~~~~~~~

CMIP6 data are first interpolated onto the ERA5 grid and saved as Zarr files.
This step is performed once and avoids repeating the interpolation during the
bias-correction step.

The preprocessing step includes:

- loading ERA5 and CMIP6 NetCDF files,
- renaming CMIP6 coordinates when needed,
- adding a cyclic longitude point,
- interpolating CMIP6 data onto the ERA5 grid,
- writing the interpolated CMIP6 data to Zarr.

2. Bias correction
~~~~~~~~~~~~~~~~~~

The bias-correction step reads:

- ERA5 training data from NetCDF files,
- preprocessed CMIP6 training data from Zarr,
- preprocessed CMIP6 application data from Zarr.

Quantile Mapping is then fitted on the training period and applied to the target
application period. The correction is performed by spatial chunks over latitude
and longitude to limit memory usage.

The resulting corrected CMIP6 data are saved as NetCDF files and can be used as
inputs for later downscaling experiments.

Chunked processing
------------------

AID-BC uses chunking for two different purposes.

During preprocessing, chunking controls how the interpolated CMIP6 data are
stored on disk in Zarr format. This does not change the data values.

During bias correction, chunking controls the spatial blocks processed by
Quantile Mapping. For each spatial block, all available time steps are used.

This design allows the package to process large global datasets without loading
the full spatial domain into memory at once.

Current method
--------------

The current bias-correction method is univariate Quantile Mapping. For each
spatial chunk, the method learns the statistical relationship between CMIP6 and
ERA5 over a training period, then applies the correction to a target CMIP6
application period.

This correction step is intended to reduce systematic distributional biases in
CMIP6 before the data are used by AI downscaling models.

Planned extensions
------------------

Future developments will focus on multivariate bias correction. Unlike
univariate methods, multivariate approaches aim to correct several variables
jointly and to better preserve dependencies between variables, temporal
structures, and spatial patterns.

Optimal Transport-based methods are planned as a major direction for these
extensions. They are expected to provide a framework for mapping distributions
between biased model outputs and reference data while accounting for
multidimensional dependencies.

These developments are intended to make AID-BC suitable for preparing more
physically consistent inputs for AI-based downscaling models.

Typical use case
----------------

A typical use case is:

- train Quantile Mapping using historical CMIP6 and ERA5 data,
- apply the trained correction to a future CMIP6 year,
- save the corrected field as a NetCDF file,
- use the corrected CMIP6 field as input to an AI downscaling pipeline.

For example, the workflow can be used to train on CMIP6 historical data from
2000 to 2014, apply the correction to a future CMIP6 simulation for 2021, and
then use the corrected 2021 field for downstream downscaling experiments.
