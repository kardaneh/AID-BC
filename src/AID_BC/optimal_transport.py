# Copyright 2026 The swirl_dynamics Authors.

# Modifications Copyright 2026 IPSL / CNRS / Sorbonne University
# Modifications by Kishanthan Kingston.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
State = tuple[Array, Array, Array, Array]

# We need to enable x64 to avoid numerical issues.
# Do not force x64 here.
# The dtype is selected by OptimalTransportCorrector.
jax.config.update("jax_enable_x64", True)


# TODO: Enable dataclasses for SinkhornOutput.
# @jax.tree_util.register_dataclass
# @dataclasses.dataclass(kw_only=True)
class SinkhornOutput(NamedTuple):
    """
    Output of the Sinkhorn optimal transport solver.

    Attributes
    ----------
    potentials : tuple of jax.Array
        Dual potentials associated with the source and target distributions.
    cost_matrix : jax.Array
        Pairwise transport cost matrix.
    epsilon : float
        Entropic regularization parameter.
    reg_ot_cost : jax.Array or None
        Regularized optimal transport cost.
    threshold : float or None
        Convergence threshold used by the Sinkhorn solver.
    converged : bool or None
        Whether the solver converged according to the configured threshold.
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
    # TODO: Compute deviation from optimality in the solutions.
    # errors: jnp.ndarray | None = None

    @property
    def fu(self) -> Array:
        """The first dual potential."""
        return self.potentials[0]

    @property
    def gv(self) -> Array:
        """The second dual potential."""
        return self.potentials[1]

    @property
    def cost(self) -> Array:
        """The cost matrix."""
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

    The algorithm is implemented in log space using log-sum-exp operations
    for improved numerical stability.

    Parameters
    ----------
    epsilon : float
        Entropic regularization parameter.
    sharding : jax.sharding.Sharding or None, optional
        Optional JAX sharding specification for the cost matrix.
    num_iterations : int, default=100
        Maximum number of Sinkhorn iterations.
    threshold : float, default=1e-3
        Convergence threshold based on changes in the dual potentials.
    metric : callable, optional
        Element-wise cost function. The default is the squared difference.
    eps_marginal : float, default=1e-12
        Small positive value added to marginal densities before taking
        logarithms.

    Attributes
    ----------
    epsilon : float
        Entropic regularization parameter.
    num_iterations : int
        Maximum number of Sinkhorn iterations.
    metric : callable
        Element-wise cost function.
    sharding : jax.sharding.Sharding or None
        Optional sharding specification.
    threshold : float
        Convergence threshold.
    eps_marginal : float
        Numerical stabilization constant for marginal densities.
    """

    def __init__(
        self,
        epsilon: float,
        sharding: jax.sharding.Sharding | None = None,
        num_iterations: int = 100,
        threshold: float = 1e-3,
        metric: Callable[[Array], Array] = lambda x: jnp.pow(x, 2),  # x²
        eps_marginal: float = 1e-12,
    ):
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.metric = metric
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
        """
        return self._solver(x, y)

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

        Raises
        ------
        ValueError
            If x or y is not two-dimensional.
        """
        # We assume that x and y are 2d arrays. First dimension is the number of
        # points, and second dimension is the feature dimension.
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "x and y should be 2d arrays, instead their dimensions are"
                f" {x.ndim}, {y.ndim} respectively."
            )

        num_x = x.shape[0]
        num_y = y.shape[0]

        # Defines the marginal densities as empirical measures.
        a, b = jnp.ones((num_x,)) / num_x, jnp.ones((num_y,)) / num_y

        # Initialises approximation vectors in log domain.
        u, v = jnp.zeros_like(a), jnp.zeros_like(b)

        # Computes cost matrix.
        cost_matrix = self._compute_cost(x, y)
        # Adds sharding constraints, if sharding is provided.
        if self.sharding is not None:
            cost_matrix = jax.lax.with_sharding_constraint(cost_matrix, self.sharding)

        jax.debug.print(
            "[OT] cost min={} mean={} max={}",
            jnp.min(cost_matrix),
            jnp.mean(cost_matrix),
            jnp.max(cost_matrix),
        )

        # Defines the body function for the while loop.
        def body_fun(val: tuple[int, State]) -> tuple[int, State]:
            # Unpacks the state, and iteration count.
            i, (_, _, u, v) = val
            # Saves the initial values of the potentials.
            u0, v0 = u, v

            # Updates the u potential following: u^{l+1} = a / (K v^l) in log space.
            kernel_matrix = self._log_gibbs_kernel(u, v, cost_matrix)
            u_ = jnp.log(a + self.eps_marginal) - jax.scipy.special.logsumexp(
                kernel_matrix, axis=1
            )
            u = self.epsilon * u_ + u

            # Updates the v potential following: v^{l+1} = b / (K^T u^(l+1)) in log
            # space.
            kernel_matrix_t = self._log_gibbs_kernel(u, v, cost_matrix).T
            v_ = jnp.log(b + self.eps_marginal) - jax.scipy.special.logsumexp(
                kernel_matrix_t, axis=1
            )
            v = self.epsilon * v_ + v

            error = jnp.linalg.norm(u0 - u) + jnp.linalg.norm(v0 - v)

            def print_progress(_):
                jax.debug.print(
                    "[OT] iteration={}/{} error={}",
                    i + 1,
                    self.num_iterations,
                    error,
                )
                return 0

            jax.lax.cond(
                ((i + 1) % 100) == 0,
                print_progress,
                lambda _: 0,
                operand=0,
            )

            return (i + 1, (u0, v0, u, v))

        # Condition function for stopping the while loop.
        def cond_fun(val: tuple[int, State]) -> Array:
            # Unpacks the state, and iteration count.
            i, (u0, v0, u, v) = val

            return jnp.logical_and(
                (jnp.linalg.norm(u0 - u) + jnp.linalg.norm(v0 - v)) > self.threshold,
                i < self.num_iterations,
            )

        # Runs the loop of the iteration.
        num_its, (u_previous, v_previous, u, v) = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=(0, (jnp.ones_like(u), jnp.ones_like(v), u, v)),
        )

        final_error = jnp.linalg.norm(u_previous - u) + jnp.linalg.norm(v_previous - v)

        jax.debug.print(
            "[OT] iterations={} final_error={} threshold={}",
            num_its,
            final_error,
            self.threshold,
        )

        # Computes transport plan pi = diag(a)*K*diag(b).
        kernel_matrix = self._log_gibbs_kernel(u, v, cost_matrix)
        pi = jnp.exp(kernel_matrix)

        # Sinkhorn distance.
        reg_ot_cost = jnp.sum(pi * cost_matrix, axis=(-2, -1))

        # Checks if the iteration converged.
        # converged = num_its < self.num_iterations - 1
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

    # def _compute_cost(self, x: Array, y: Array) -> Array:
    #  r"""Computes the cost matrix of the Sinkhorn iteration.

    #  Args:
    #    x: Collection of points of set A.
    #    y: Collection of point of set B.

    #  Returns:
    #    A matrix C of size (num_x, num_y) corresponding to the distance between
    #    each pair of points, namely:

    #    C_{i,j} = distance(x_i, y_j).

    #    Here the metric is defined as the sum over the feature dimension of a
    #    user-prescribed metric. Basically,

    #    distance(x_i, y_j) = sum_{k=1}^d self.metric(x_{i,k} - y_{j,k}),

    #    where k is the index of the feature dimension.
    #  """
    #  x_ = x[:, None, :]
    #  y_ = y[None, :, :]
    #  cost_matrix = jnp.sum(self.metric(x_ - y_), axis=-1) # self.metric(x_ - y_) = (x_ - y_)²,
    #  # metric = lambda x: jnp.pow(x, 2)
    #  return cost_matrix

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

        # Numerical rounding can produce very small negative distances.
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

        # Here we assume that the cost is the Euclidean distance. In comparison with
        # [2] we don't have a 1/2 factor in the definition of the distance, so we
        # need to divide by 2.
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
            If x and y do not have the same feature dimension.
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

        if not weights:
            num_y = y.shape[0]
            weights = jnp.ones((num_y,)) / num_y

        def _transport_direct(x: Array) -> Array:
            # The dimension should be (1, num_y)
            cost = jnp.squeeze(self._compute_cost(x, y))
            z = jnp.exp((potential - cost) / self.epsilon) * weights
            # return jnp.sum(y * z[:, None], axis=-1)/jnp.sum(z)
            return jnp.sum(y * z[:, None], axis=0) / jnp.sum(z)

        return _transport_direct
