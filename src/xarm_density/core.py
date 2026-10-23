"""Shared quadratic-basis CDF-QP utilities."""

from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from xarm_density.qp import CDFQP, CDFQPConfig, CDFQPData
from xarm_density.density import smoothstep_cinfty

jax.config.update("jax_enable_x64", True)


def quadratic_basis(x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Total-degree quadratic monomials in an arbitrary state dimension."""

    z = x / scale
    terms = [jnp.asarray(1.0)]
    terms.extend(z[i] for i in range(z.shape[0]))
    terms.extend(z[i] * z[j] for i in range(z.shape[0]) for j in range(i, z.shape[0]))
    return jnp.stack(terms)


def block_input_map(values: np.ndarray, rho: float, channels: int, density_floor=1e-10) -> np.ndarray:
    """Map channel-stacked weighted-flux coefficients to the applied input."""

    row = np.asarray(values, dtype=float) / max(float(rho), density_floor)
    return np.kron(np.eye(channels), row[None, :])


def pointwise_transport_row(basis_jacobian: np.ndarray) -> np.ndarray:
    """Stack d w_j / d x_j for channel-wise copies of one scalar basis."""

    jac = np.asarray(basis_jacobian, dtype=float)
    return np.concatenate([jac[:, channel] for channel in range(jac.shape[1])])


def inverse_bump_from_clearance(clearance: jnp.ndarray, sensing_width: float) -> jnp.ndarray:
    """C-infinity factor that is zero at collision and one outside sensing."""

    return smoothstep_cinfty(clearance / sensing_width)


class QuadraticCDFQP:
    """Fixed-size cached Clarabel QP for one pointwise CDF constraint."""

    def __init__(
        self,
        n_theta: int,
        n_inputs: int,
        residual_margin: float,
        input_bound: float,
        solver_settings: dict | None = None,
    ) -> None:
        self.controller = CDFQP(
            n_theta=n_theta,
            n_constraints=1,
            n_inputs=n_inputs,
            config=CDFQPConfig(
                residual_margin=residual_margin,
                input_lower=-input_bound,
                input_upper=input_bound,
                solver_settings=solver_settings or {},
            ),
        )

    def solve(
        self,
        input_map: np.ndarray,
        transport_row: np.ndarray,
        nominal_input: np.ndarray,
        task_map: np.ndarray | None = None,
        task_weight: float = 1.0,
        input_weight: float = 1e-2,
        theta_weight: float = 1e-5,
    ) -> dict[str, object]:
        if task_map is None:
            task_map = input_map
        n_theta = input_map.shape[1]
        hessian = (
            task_weight * task_map.T @ task_map
            + input_weight * input_map.T @ input_map
            + theta_weight * np.eye(n_theta)
        )
        linear_cost = -task_weight * task_map.T @ np.asarray(nominal_input, dtype=float)
        result = self.controller.solve(
            CDFQPData(
                hessian=hessian,
                linear_cost=linear_cost,
                cdf_matrix=np.asarray(transport_row, dtype=float)[None, :],
                drift=np.zeros(1),
                input_matrix=np.asarray(input_map, dtype=float),
            )
        )
        return {
            "theta": result.theta,
            "input": result.control,
            "residual": float(result.residuals[0]),
            "status": result.status,
            "solve_time": result.solve_time,
            "solver_solve_time": result.solver_solve_time,
            "solver_num_iters": result.solver_num_iters,
        }


class ReducedQuadraticCDFQP(QuadraticCDFQP):
    """Exact row-space reduction for six commands and one transport constraint.

    Components of theta orthogonal to [B; a] affect neither commands nor the
    constraint. With positive isotropic coefficient regularization their
    optimal value is zero. An orthonormal QR basis therefore preserves the QP.
    """

    def __init__(self, n_theta, n_inputs, residual_margin, input_bound, solver_settings=None):
        self.full_dimension = n_theta
        self.reduced_dimension = min(n_theta, n_inputs + 1)
        super().__init__(self.reduced_dimension, n_inputs, residual_margin, input_bound, solver_settings)

    def solve(self, input_map, transport_row, nominal_input, task_map=None,
              task_weight=1.0, input_weight=1e-2, theta_weight=1e-5):
        start = perf_counter()
        if theta_weight <= 0:
            raise ValueError("Row-space reduction requires positive theta regularization")
        rows = np.vstack((input_map, np.asarray(transport_row)[None, :]))
        # Normalize rows for conditioning without changing their span.
        rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)
        basis, _ = np.linalg.qr(rows.T, mode="reduced")
        task = input_map if task_map is None else task_map
        reduced_task = task @ basis
        if not np.allclose(task, reduced_task @ basis.T, rtol=1e-8, atol=1e-10):
            raise ValueError("Task objective lies outside the command/transport row space")
        result = super().solve(
            input_map @ basis, np.asarray(transport_row) @ basis,
            nominal_input, reduced_task, task_weight, input_weight, theta_weight,
        )
        result["theta"] = basis @ result["theta"]
        result["input"] = input_map @ result["theta"]
        result["residual"] = float(np.asarray(transport_row) @ result["theta"])
        result["solve_time"] = perf_counter() - start
        return result
