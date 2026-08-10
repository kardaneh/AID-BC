Optimal Transport
=================

Optimal Transport is a bias-correction method used in AID-BC to reduce
systematic distributional differences between CMIP6 climate model outputs and
ERA5 reference data. Unlike Quantile Mapping, it can correct several climate
variables and every spatial grid point jointly, rather than one variable at a
time.

The method is fitted over a selected training period and then applied to a
CMIP6 application period. The corrected CMIP6 fields can subsequently be used
as large-scale inputs for AI-based downscaling experiments.

Purpose
-------

Climate model outputs often contain systematic biases compared with reference
datasets or reanalysis products.

Optimal Transport corrects these biases by finding a mapping that transports
the biased CMIP6 distribution onto the ERA5 reference distribution at minimum
cost. Because this mapping can act on several variables and grid points at once, it
can account for spatial and inter-variable dependencies that are not explicitly
represented by a variable-by-variable correction.

General principle
-----------------

Entropic regularization and the Sinkhorn algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AID-BC treats each time step as a single point in a feature space (see
`Feature space construction`_ below), and computes a transport plan between
the empirical distribution of CMIP6 points and the empirical distribution of
ERA5 points. The cost of moving a CMIP6 point ``x`` to an ERA5 point ``y`` is
the squared Euclidean distance in this feature space:

.. math::

   c(x, y) = \|x - y\|^2

Directly solving for the optimal transport plan is computationally expensive.
AID-BC instead solves the entropy-regularized problem:

.. math::

   \min_{\pi \in \Pi(a, b)} \sum_{i,j} \pi_{ij}\, c(x_i, y_j)
   \;+\; \varepsilon \sum_{i,j} \pi_{ij} \left( \log \pi_{ij} - 1 \right)

where ``a`` and ``b`` are the empirical marginal weights associated with the
CMIP6 and ERA5 samples, and :math:`\varepsilon` controls the strength of the
entropic regularization.

This regularized problem is solved with the Sinkhorn algorithm, which
alternately updates two dual potentials ``u`` and ``v`` (one per CMIP6 sample,
one per ERA5 sample) in log space until they stop changing significantly. The
iteration stops when the combined change in ``u`` and ``v`` falls below a
convergence threshold, or when a maximum number of iterations is reached.

Transport map
~~~~~~~~~~~~~

After fitting, AID-BC uses the fitted ERA5-side dual potential and target
samples to construct a transport function for new CMIP6 samples.

During application, a new Sinkhorn problem is not solved for each CMIP6 sample.
Instead, the transport map is evaluated directly using the formulation of
Proposition 2 in Pooladian and Niles-Weed [1]:

.. math::

   f_{\varepsilon}(x) = -\varepsilon \log \left(
       \sum_i \exp\!\left( \frac{g_{\varepsilon}(y_i) - c(x, y_i)}{\varepsilon} \right) b_i
   \right)

.. math::

   T(x) = x - \tfrac{1}{2} \nabla f_{\varepsilon}(x)

where :math:`g_{\varepsilon}` is the fitted ERA5-side potential. This gives a
smooth, differentiable transport function that can be evaluated on any new
CMIP6 sample without recomputing the full coupling, which is what AID-BC uses
during the application step.

Feature space construction
--------------------------

Before fitting, ERA5 and CMIP6 training fields can be standardized using a
shared per-feature normalization.
By default, the normalization parameters are estimated jointly from the ERA5
and historical CMIP6 training samples. The same parameters are used during
application, and the transported values are transformed back to their original
physical scale afterwards.

The reshaping of the fields depends on how many variables are corrected
jointly:

- For a single variable, each time step is reshaped from
  ``(latitude, longitude)`` to a vector of length ``latitude * longitude``.
- For several variables corrected jointly, each variable's grid is flattened
  the same way and the variables are concatenated into a single vector per
  time step, of length ``n_variables * latitude * longitude``. All variables
  and grid points are therefore transported together as one multivariate
  point.

Training and application periods
--------------------------------

The Optimal Transport workflow separates the data into two periods, following
the same convention as Quantile Mapping.

1. Training period
~~~~~~~~~~~~~~~~~~

The training period is used to fit the transport map between ERA5 and
historical CMIP6 data.

Both datasets must be represented on the same latitude-longitude grid. In the
current AID-BC workflow, correction is performed on the native CMIP6 grid.
ERA5 data are therefore either prepared onto the CMIP6 grid during the
workflow or loaded from previously prepared ERA5-on-CMIP6 files.

2. Application period
~~~~~~~~~~~~~~~~~~~~~

The application period contains the CMIP6 data to be corrected using the
fitted transport map.

It can correspond to:

- a historical period not used during training,
- a validation or test period,
- a future CMIP6 simulation period.

Corrected fields are reconstructed on the native CMIP6 grid and saved as
yearly NetCDF files, one per corrected variable.

Current implementation
----------------------

AID-BC implements Optimal Transport as a method that accepts either:

- exactly one climate variable (univariate correction), or
- several climate variables corrected jointly (multivariate correction).

As with Quantile Mapping, AID-BC can fit either one global corrector using
all training samples, or one independent corrector for each month.

Monthly splitting is particularly useful for Optimal Transport. For ``n_x``
historical CMIP6 samples and ``n_y`` ERA5 reference samples, the Sinkhorn
problem involves a pairwise cost matrix of shape ``(n_x, n_y)``. When both
datasets contain a similar number of samples, this results in approximately
quadratic scaling with the number of training samples.

Fitting one corrector per month therefore substantially reduces the size of
each Optimal Transport problem.

Solver parameters
~~~~~~~~~~~~~~~~~

The Sinkhorn solver is controlled by the following parameters:

``epsilon``
   Entropic regularization strength relative to the scale of the normalized
   squared-distance cost. Larger values produce a more strongly regularized
   and smoother transport problem. Smaller values reduce the amount of
   entropic regularization but generally make Sinkhorn convergence more
   difficult.

``num_iterations``
   Maximum number of Sinkhorn iterations. Smaller ``epsilon`` values generally
   require more iterations to satisfy the convergence criterion.

``threshold``
   Convergence threshold on the change in the dual potentials between
   iterations. This threshold must be set consistently with ``epsilon`` and
   the number of training samples: an overly strict threshold may never be
   reached within a practical iteration budget for small ``epsilon``, causing
   the fit to fail even though the transport map is otherwise usable.

``dtype``
   Floating-point precision used by the solver (``float32`` or ``float64``).

``batch_size``
   Number of time steps transported together when applying the fitted map,
   to control memory usage during application.

Limitations
-----------

Optimal Transport in AID-BC is subject to the following limitations.

Entropic smoothing
~~~~~~~~~~~~~~~~~~

The entropic regularization needed to make Sinkhorn tractable introduces a
smoothing bias in the transport plan. If ``epsilon`` is too large relative to
the scale of the cost function, the corrected distribution can under-estimate
variance and extreme values. In multivariate mode, a single ``epsilon`` is
shared across all jointly transported variables and grid points; variables
with different intrinsic variability (for example, wind components compared
to temperature) can therefore be smoothed to different degrees by the same
``epsilon``, and this trade-off should be checked per variable rather than
assumed to be uniform.

Convergence sensitivity
~~~~~~~~~~~~~~~~~~~~~~~

Smaller ``epsilon`` values, which are often needed to preserve variance,
degrade the convergence rate of the Sinkhorn algorithm. Reaching a strict
convergence threshold can require a large number of iterations, and the
required budget can vary noticeably between training periods (for example,
between months).

Stationarity assumption
~~~~~~~~~~~~~~~~~~~~~~~

As with Quantile Mapping, the method assumes that the statistical
relationship between CMIP6 and ERA5 estimated during the training period
remains applicable during the application period.

Computational cost
~~~~~~~~~~~~~~~~~~

For ``n_x`` CMIP6 source samples and ``n_y`` ERA5 target samples, the dominant
pairwise matrices have shape ``(n_x, n_y)``. Memory usage therefore scales as
:math:`O(n_x n_y)`. When both sample counts are approximately equal to
:math:`N`, this becomes approximately :math:`O(N^2)`.

Adding more jointly transported variables increases the feature dimension and
the cost of computing pairwise distances, but does not change the
``n_x × n_y`` dimensions of the Sinkhorn matrices.

References
----------

[1] Pooladian, Aram-Alexandre, and Niles-Weed, Jonathan. "Entropic estimation of
optimal transport maps." arXiv preprint arXiv:2109.12004 (2021).
