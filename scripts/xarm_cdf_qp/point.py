"""Three-dimensional point/end-effector sphere-world CDF-QP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xarm_cdf_qp.core import (
    QuadraticCDFQP,
    block_input_map,
    inverse_bump_from_clearance,
    pointwise_transport_row,
    quadratic_basis,
)


@dataclass(frozen=True)
class PointWorldConfig:
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    obstacle_centers: tuple[tuple[float, float, float], ...] = ()
    obstacle_radii: tuple[float, ...] = ()
    body_radius: float = 0.04
    sensing_width: float = 0.16
    alpha: float = 0.2
    v_epsilon: float = 1e-3
    basis_scale: float = 0.6
    residual_margin: float = 0.02
    speed_limit: float = 0.35
    nominal_gain: float = 1.2
    dt: float = 0.02
    steps: int = 600
    target_radius: float = 0.025


class PointSphereCDFController:
    def __init__(self, config: PointWorldConfig):
        self.config = config
        centers = jnp.asarray(config.obstacle_centers, dtype=jnp.float64).reshape((-1, 3))
        radii = jnp.asarray(config.obstacle_radii, dtype=jnp.float64)
        goal = jnp.asarray(config.goal, dtype=jnp.float64)
        scale = jnp.full(3, config.basis_scale)

        def psi(x: jnp.ndarray) -> jnp.ndarray:
            if centers.shape[0] == 0:
                return jnp.asarray(1.0)
            clearance = (
                jnp.linalg.norm(x[None, :] - centers, axis=1)
                - radii
                - config.body_radius
            )
            return jnp.prod(inverse_bump_from_clearance(clearance, config.sensing_width))

        def rho(x: jnp.ndarray) -> jnp.ndarray:
            value = jnp.dot(x - goal, x - goal) + config.v_epsilon
            return psi(x) / value**config.alpha

        def flux_basis(x: jnp.ndarray) -> jnp.ndarray:
            return psi(x) * quadratic_basis(x, scale)

        self.psi = jax.jit(psi)
        self.rho = jax.jit(rho)
        self.flux_basis = jax.jit(flux_basis)
        self.flux_basis_jacobian = jax.jit(jax.jacfwd(flux_basis))
        n_terms = 1 + 3 + 3 * 4 // 2
        self.qp = QuadraticCDFQP(
            n_theta=3 * n_terms,
            n_inputs=3,
            residual_margin=config.residual_margin,
            input_bound=config.speed_limit,
        )

    def nominal_velocity(self, x: np.ndarray) -> np.ndarray:
        error = np.asarray(self.config.goal) - x
        velocity = self.config.nominal_gain * error
        norm = np.linalg.norm(velocity)
        if norm > self.config.speed_limit:
            velocity *= self.config.speed_limit / norm
        return velocity

    def solve(self, x: np.ndarray) -> dict[str, Any]:
        xj = jnp.asarray(x)
        rho = float(self.rho(xj))
        basis = np.asarray(self.flux_basis(xj))
        basis_jacobian = np.asarray(self.flux_basis_jacobian(xj))
        input_map = block_input_map(basis, rho, channels=3)
        transport = pointwise_transport_row(basis_jacobian)
        nominal = self.nominal_velocity(x)
        result = self.qp.solve(
            input_map=input_map,
            transport_row=transport,
            nominal_input=nominal,
            task_weight=1.0,
            input_weight=1e-2,
            theta_weight=2e-4,
        )
        result.update({"rho": rho, "psi": float(self.psi(xj)), "nominal": nominal})
        return result


def minimum_clearance(x: np.ndarray, config: PointWorldConfig) -> float:
    if not config.obstacle_centers:
        return float("inf")
    return float(
        min(
            np.linalg.norm(x - np.asarray(center)) - radius - config.body_radius
            for center, radius in zip(config.obstacle_centers, config.obstacle_radii)
        )
    )


def simulate_point_world(config: PointWorldConfig) -> dict[str, Any]:
    controller = PointSphereCDFController(config)
    x = np.asarray(config.start, dtype=float)
    states = [x.copy()]
    controls = []
    nominal_controls = []
    residuals = []
    clearances = []
    densities = []
    statuses = []
    solve_times = []
    solver_times = []

    controller.solve(x)
    for _ in range(config.steps):
        if np.linalg.norm(x - np.asarray(config.goal)) <= config.target_radius:
            break
        result = controller.solve(x)
        velocity = np.asarray(result["input"], dtype=float)
        x = x + config.dt * velocity
        states.append(x.copy())
        controls.append(velocity)
        nominal_controls.append(np.asarray(result["nominal"]))
        residuals.append(float(result["residual"]))
        clearances.append(minimum_clearance(x, config))
        densities.append(float(result["rho"]))
        statuses.append(str(result["status"]))
        solve_times.append(float(result["solve_time"]))
        solver_times.append(float(result["solver_solve_time"]))

    states_array = np.asarray(states)
    controls_array = np.asarray(controls)
    final_error = float(np.linalg.norm(states_array[-1] - np.asarray(config.goal)))
    finite_clearances = np.asarray(clearances)
    return {
        "states": states_array,
        "controls": controls_array,
        "nominal_controls": np.asarray(nominal_controls),
        "residuals": np.asarray(residuals),
        "clearances": finite_clearances,
        "densities": np.asarray(densities),
        "statuses": statuses,
        "solve_times": np.asarray(solve_times),
        "solver_times": np.asarray(solver_times),
        "metrics": {
            "steps": int(controls_array.shape[0]),
            "target_reached": bool(final_error <= config.target_radius),
            "final_target_error": final_error,
            "minimum_clearance": (
                float(np.min(finite_clearances))
                if config.obstacle_centers and finite_clearances.size
                else None
            ),
            "minimum_cdf_residual": float(np.min(residuals)) if residuals else None,
            "maximum_speed": (
                float(np.max(np.linalg.norm(controls_array, axis=1)))
                if controls_array.size
                else 0.0
            ),
            "all_qps_feasible": bool(all(status.startswith("optimal") for status in statuses)),
            "mean_solver_time_ms": (
                float(1e3 * np.mean(solver_times)) if solver_times else 0.0
            ),
            "mean_controller_qp_time_ms": (
                float(1e3 * np.mean(solve_times)) if solve_times else 0.0
            ),
        },
    }
