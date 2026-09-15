# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# ============================================================================
# ORIGINAL WORK (SWIRL DYNAMICS / GOOGLE)
# ============================================================================
# This work is a derivative of the Sinkhorn optimal transport implementation
# from the swirl_dynamics project developed by the swirl_dynamics Authors.
#
# Original work: Copyright 2026 The swirl_dynamics Authors.
# Original license: Apache License, Version 2.0.
# Original source: https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/debiasing/optimal_transport/sinkhorn.py
#
# ============================================================================
# MODIFICATIONS AND ADDITIONS (IPSL / CNRS / Sorbonne University)
# ============================================================================
# Modifications include:
#
#   1. Optimized squared Euclidean cost computation
#      - Replaced the broadcasted pairwise difference computation with
#        ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
#      - Avoided materializing an intermediate
#        (n_source, n_target, n_features) array
#      - Reduced peak memory usage for pairwise cost computation
#      - Specialized the solver to squared Euclidean transport cost
#
#   2. Optimized Sinkhorn iteration
#      - Precomputed -cost_matrix / epsilon before the Sinkhorn loop
#      - Precomputed logarithms of the source and target marginal densities
#      - Reused iteration-invariant quantities throughout the fixed-point loop
#
#   3. Simplified convergence-state handling
#      - Simplified the while_loop state to (iteration, u, v, error)
#      - Carried the convergence error explicitly in the loop state
#      - Initialized the convergence error to +inf to guarantee at least one
#        Sinkhorn iteration
#
#   4. Improved convergence reporting
#      - Replaced the iteration-count-based convergence flag with an explicit
#        comparison between the final convergence error and the configured
#        threshold
#
#   5. Added input and parameter validation
#      - Added validation for epsilon, num_iterations, threshold, and
#        eps_marginal
#      - Added validation for input dimensionality, feature compatibility,
#        and empty point clouds
#      - Moved public input validation outside the jitted solver
#
#   6. Improved documentation and code clarity
#      - Converted documentation to NumPy-style docstrings
#      - Updated parameter and return-value descriptions
#      - Improved comments and variable naming for clarity
#      - Removed the generic metric parameter after specializing the solver
#        to squared Euclidean cost
#
# ============================================================================
# LICENSE
# ============================================================================
# The original swirl_dynamics code is licensed under the Apache License,
# Version 2.0, and remains subject to the terms and conditions of that license.
#
# The modifications and additions made by IPSL / CNRS / Sorbonne University
# are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License, to the extent permitted by the Apache License,
# Version 2.0.
#
# Apache License, Version 2.0:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0:
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# ============================================================================
# ACKNOWLEDGMENTS
# ============================================================================
# We thank the swirl_dynamics Authors for developing and releasing the original
# Sinkhorn optimal transport implementation under an open-source license that
# enables further research and development.

"""
Sinkhorn algorithm for optimal transport.

References:

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal
transport." Advances in neural information processing systems 26 (2013).

[2] Pooladian, Aram-Alexandre, and Niles-Weed, Jonathan. "Entropic estimation of
optimal transport maps." arXiv preprint arXiv:2109.12004 (2021).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array

# We need to enable x64 to avoid numerical issues.
jax.config.update("jax_enable_x64", True)


class SinkhornOutput(NamedTuple):
    """
    Output of the Sinkhorn optimal transport solver.

    Attributes
    ----------
    potentials : tuple[jax.Array, jax.Array]
        Dual potentials associated with the source and target distributions.
    cost_matrix : jax.Array
        Pairwise squared-Euclidean cost matrix.
    epsilon : float
        Entropic regularization parameter.
    reg_ot_cost : jax.Array or None
        Regularized optimal transport cost.
    threshold : float or None
        Convergence threshold used by the Sinkhorn solver.
    converged : bool or None
        Whether the final change in the dual potentials is below threshold.
    num_iterations : int or None
        Number of Sinkhorn iterations performed.
    """

    potentials: tuple[Array, Array]
    cost_matrix: Array
    epsilon: float
    reg_ot_cost: Array | None = None
    threshold: float | None = None
    converged: bool | None = None
    num_iterations: int | None = None

    @property
    def fu(self) -> Array:
        """Return the source dual potential."""
        return self.potentials[0]

    @property
    def gv(self) -> Array:
        """Return the target dual potential."""
        return self.potentials[1]

    @property
    def cost(self) -> Array:
        """Return the pairwise cost matrix."""
        return self.cost_matrix

    @property
    def transport_plan(self) -> Array:
        """
        Compute the transport plan from the dual potentials.

        Returns
        -------
        jax.Array
            Entropic transport plan.

        Raises
        ------
        ValueError
            If epsilon is not strictly positive.
        """
        if self.epsilon <= 0:
            raise ValueError("Epsilon should be positive.")
        kernel = -self.cost + self.fu[:, None] + self.gv[None, :]
        kernel /= self.epsilon

        return jnp.exp(kernel)


class SinkhornSolver:
    """
    Entropy-regularized optimal transport solver using Sinkhorn iterations.

    The solver operates in log space and uses "logsumexp" for numerical
    stability. The pairwise cost is the squared Euclidean distance.

    Parameters
    ----------
    epsilon : float
        Entropic regularization parameter. Must be strictly positive.
    sharding : jax.sharding.Sharding or None, optional
        Optional JAX sharding specification for the cost matrix.
    num_iterations : int, default=100
        Maximum number of Sinkhorn iterations.
    threshold : float, default=1e-3
        Convergence threshold based on changes in the dual potentials.
    eps_marginal : float, default=1e-12
        Small positive value added to marginal densities before taking
        logarithms.
    """

    def __init__(
        self,
        epsilon: float,
        sharding: jax.sharding.Sharding | None = None,
        num_iterations: int = 100,
        threshold: float = 1e-3,
        eps_marginal: float = 1e-12,
    ):
        if epsilon <= 0:
            raise ValueError(
                f"epsilon must be strictly positive, got epsilon={epsilon}."
            )
        if num_iterations <= 0:
            raise ValueError(
                "num_iterations must be strictly positive, got "
                f"num_iterations={num_iterations}."
            )
        if threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got threshold={threshold}."
            )
        if eps_marginal < 0:
            raise ValueError(
                "eps_marginal must be non-negative, got "
                f"eps_marginal={eps_marginal}."
            )

        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.sharding = sharding
        self.threshold = threshold
        self.eps_marginal = eps_marginal
        self._solver = jax.jit(self._forward_solve)

    def __call__(self, x: Array, y: Array) -> SinkhornOutput:
        """
        Solve the optimal transport problem between two point clouds.

        Parameters
        ----------
        x : jax.Array
            Source samples with shape (n_source, n_features).
        y : jax.Array
            Target samples with shape (n_target, n_features).

        Returns
        -------
        SinkhornOutput
            Sinkhorn solution containing dual potentials, cost matrix,
            regularized transport cost, convergence state, and iteration count.

        Raises
        ------
        ValueError
            If the inputs are not two-dimensional, have incompatible feature
            dimensions, or contain no samples.
        """
        self._validate_inputs(x, y)
        return self._solver(x, y)

    @staticmethod
    def _validate_inputs(x: Array, y: Array) -> None:
        """
        Validate source and target point clouds.

        Parameters
        ----------
        x : jax.Array
            Source samples.
        y : jax.Array
            Target samples.

        Raises
        ------
        ValueError
            If the arrays are not two-dimensional, their feature dimensions do
            not match, or either point cloud is empty.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "x and y must be 2D arrays with shape "
                "(n_samples, n_features); got "
                f"x.ndim={x.ndim} and y.ndim={y.ndim}."
            )
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "x and y must have the same feature dimension; got "
                f"x.shape={x.shape} and y.shape={y.shape}."
            )
        if x.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(
                "x and y must each contain at least one sample; got "
                f"x.shape={x.shape} and y.shape={y.shape}."
            )

    def _forward_solve(self, x: Array, y: Array) -> SinkhornOutput:
        """
        Compute the entropy-regularized optimal transport solution.

        Parameters
        ----------
        x : jax.Array
            Source samples with shape (n_source, n_features).
        y : jax.Array
            Target samples with shape (n_target, n_features).

        Returns
        -------
        SinkhornOutput
            Solver output containing the dual potentials, cost matrix,
            convergence information, and regularized transport cost.
        """
        num_x = x.shape[0]
        num_y = y.shape[0]

        # Defines the marginal densities as empirical measures
        a, b = jnp.ones((num_x,)) / num_x, jnp.ones((num_y,)) / num_y

        # Dual potentials in log-domain form
        u, v = jnp.zeros_like(a), jnp.zeros_like(b)

        # Pairwise squared-Euclidean cost matrix
        cost_matrix = self._compute_cost(x, y)
        # Adds sharding constraints, if sharding is provided
        if self.sharding is not None:
            cost_matrix = jax.lax.with_sharding_constraint(cost_matrix, self.sharding)

        # These quantities are invariant across Sinkhorn iterations. Hoisting
        # them out of the loop avoids repeated matrix-scale divisions and logs.
        neg_cost_scaled = -cost_matrix / self.epsilon
        neg_cost_scaled_t = neg_cost_scaled.T
        log_a = jnp.log(a + self.eps_marginal)
        log_b = jnp.log(b + self.eps_marginal)

        # Defines the body function for the while loop.
        def body_fun(
            val: tuple[int, Array, Array, Array],
        ) -> tuple[int, Array, Array, Array]:
            """Perform one Sinkhorn fixed-point iteration."""
            i, u, v, _ = val
            u_previous = u
            v_previous = v

            # u^{l+1} update in log space.
            kernel_matrix = (
                neg_cost_scaled + u[:, None] / self.epsilon + v[None, :] / self.epsilon
            )
            u_update = log_a - jax.scipy.special.logsumexp(kernel_matrix, axis=1)
            u = self.epsilon * u_update + u

            # v^{l+1} update in log space. The transposed precomputed cost avoids
            # transposing a newly constructed full kernel matrix each iteration.
            kernel_matrix_t = (
                neg_cost_scaled_t
                + v[:, None] / self.epsilon
                + u[None, :] / self.epsilon
            )
            v_update = log_b - jax.scipy.special.logsumexp(kernel_matrix_t, axis=1)
            v = self.epsilon * v_update + v

            # Compute the stopping error once and carry it in the loop state.
            error = jnp.linalg.norm(u_previous - u) + jnp.linalg.norm(v_previous - v)
            return i + 1, u, v, error

        # Condition function for stopping the while loop.
        def cond_fun(val: tuple[int, Array, Array, Array]) -> Array:
            """Continue until convergence or the iteration limit is reached."""
            i, _, _, error = val
            return jnp.logical_and(
                error > self.threshold,
                i < self.num_iterations,
            )

        # +inf guarantees that the solver performs at least one iteration.
        init_error = jnp.asarray(jnp.inf, dtype=u.dtype)
        num_its, u, v, final_error = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=(0, u, v, init_error),
        )

        # Transport plan pi = exp((-C + u + v) / epsilon)
        kernel_matrix = (
            neg_cost_scaled + u[:, None] / self.epsilon + v[None, :] / self.epsilon
        )
        pi = jnp.exp(kernel_matrix)

        # Regularized Sinkhorn transport cost
        reg_ot_cost = jnp.sum(pi * cost_matrix, axis=(-2, -1))

        # Convergence is determined directly from the final stopping error.
        converged = final_error <= self.threshold

        return SinkhornOutput(
            potentials=(u, v),
            cost_matrix=cost_matrix,
            epsilon=self.epsilon,
            num_iterations=num_its,
            reg_ot_cost=reg_ot_cost,
            converged=converged,
            threshold=self.threshold,
        )

    def _compute_cost(self, x: Array, y: Array) -> Array:
        """
        Compute the squared Euclidean distance matrix without creating
        an array of shape (n_x, n_y, n_features).

        C[i, j] = ||x[i] - y[j]||²
                = ||x[i]||² + ||y[j]||² - 2 x[i]·y[j]

        Parameters
        ----------
        x : jax.Array
            First collection of samples with shape (n_x, n_features).
        y : jax.Array
            Second collection of samples with shape (n_y, n_features).

        Returns
        -------
        jax.Array
            Cost matrix with shape (n_x, n_y).
        """
        x_squared_norm = jnp.sum(
            jnp.square(x),
            axis=1,
            keepdims=True,
        )

        y_squared_norm = jnp.sum(
            jnp.square(y),
            axis=1,
            keepdims=True,
        ).T

        cross_product = x @ y.T

        cost_matrix = x_squared_norm + y_squared_norm - 2.0 * cross_product

        # Round-off may create tiny negative values for theoretically zero
        # squared distances. Clamp only those numerical artifacts to zero.
        return jnp.maximum(cost_matrix, 0.0)

    def _log_gibbs_kernel(self, u: Array, v: Array, cost_matrix: Array) -> Array:
        """
        Computes the kernel K = diag(u) exp(-C/eps) diag(v) in log space.

        Parameters
        ----------
        u : jax.Array
            Source dual potential.
        v : jax.Array
            Target dual potential.
        cost_matrix : jax.Array
            Pairwise transport cost matrix.

        Returns
        -------
        jax.Array
            Logarithm of the rescaled Gibbs kernel.
        """
        kernel = -cost_matrix + u[:, None] + v[None, :]
        kernel /= self.epsilon
        return kernel

    def transport_fn(
        self, potential: Array, y: Array, weights: Array | None = None
    ) -> Callable[[Array], Array]:
        r"""
        Transport functions using the formulation in the proposition 2 of [2].

        We use the fact that the transport function can be written as:

        T(x) = x - 0.5 * \nabla(f_{\epsilon}(x)),

        where f_{\epsilon}(x) is the potential computed using the Eq. 9 in [2].

        Parameters
        ----------
        potential : jax.Array
            Dual potential associated with the target samples.
        y : jax.Array
            Target samples associated with the potential.
        weights : jax.Array or None, optional
            Marginal weights associated with the target samples. Uniform weights
            are used internally when needed by the potential function.

        Returns
        -------
        callable
            Function mapping source samples to transported samples.
        """

        # Computes the potential of set A.
        # f_eps = lambda x: self._potential_fn(x, potential, y, weights)
        def f_eps(x: Array) -> Array:
            return self._potential_fn(
                x,
                potential,
                y,
                weights,
            )

        # Here we assume that the cost is the Euclidean distance.
        # In comparison with [2], we do not include the 1/2 factor in the cost,
        # so the gradient term must be divided by 2.
        return jax.vmap(lambda x: x - 0.5 * jax.grad(f_eps)(x), in_axes=0, out_axes=0)

    def _potential_fn(
        self,
        x: Array,
        potential: Array,
        y: Array,
        weights: Array | None = None,
    ) -> Array:
        r"""Callback function to compute the potential.

        Here we use the formula in Proposition 2 of [2]:

        f_{\epsilon}(x) = - \epsilon \log (\sum_{i}
              exp ( g_{\epsilon}(y_i) - dist(x, y_i) ) b_i

        here b_i is the marginal density of set B (associated with y_i).

        Parameters
        ----------
        x : jax.Array
            Samples where the potential is evaluated.
        potential : jax.Array
            Dual potential associated with the target samples.
        y : jax.Array
            Target samples.
        weights : jax.Array or None, optional
            Marginal weights associated with the target samples. Uniform weights
            are used when None.

        Returns
        -------
        jax.Array
            Evaluated entropic potential.

        Raises
        ------
        ValueError
            If "x" and "y" do not have the same feature dimension.
        """
        x = jnp.atleast_2d(x)

        if weights is None:
            num_y = y.shape[0]
            weights = jnp.ones((num_y,)) / num_y

        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "x and y should have the same feature dimension, but"
                f" they have shape {x.shape[-1]}, {y.shape[-1]}, respectively."
            )

        # Computes cost matrix with respect to the current x.
        cost = jnp.squeeze(self._compute_cost(x, y))
        z = (potential - cost) / self.epsilon
        lse = -self.epsilon * jax.scipy.special.logsumexp(z, b=weights, axis=-1)
        return jnp.squeeze(lse)

    def transport_fn_direct(
        self, potential: Array, y: Array, weights: Array | None
    ) -> Callable[[Array], Array]:
        """
        Transport directly (not very stable). Using the formulas in [1].

        Parameters
        ----------
        potential : jax.Array
            One-dimensional target dual potential.
        y : jax.Array
            Target samples associated with the potential.
        weights : jax.Array or None
            Marginal weights associated with the target samples.

        Returns
        -------
        callable
            Function mapping one source sample to its transported value.

        Raises
        ------
        ValueError
            If potential is not one-dimensional or if its length differs from
            the number of target samples.
        """
        if potential.ndim != 1:
            raise ValueError(
                "The potential should be a vector, but its dimension are not one,"
                f" instead {y.ndim}"
            )
        if potential.shape[0] != y.shape[0]:
            raise ValueError(
                "We assume that the potential comes from solving Sinkhorn, but"
                f" potential.shape[0] != y.shape[0]: {potential.shape}, {y.shape}"
            )

        if weights is None:
            num_y = y.shape[0]
            weights = jnp.ones((num_y,)) / num_y

        def _transport_direct(x: Array) -> Array:
            # The dimension should be (1, num_y)
            cost = jnp.squeeze(self._compute_cost(x, y))
            z = jnp.exp((potential - cost) / self.epsilon) * weights
            # Sum over target samples (axis=0), leaving the feature dimension
            return jnp.sum(y * z[:, None], axis=0) / jnp.sum(z)

        return _transport_direct
