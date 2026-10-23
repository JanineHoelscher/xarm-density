"""UF850 joint-space CDF-QP with differentiable collision spheres."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from xarm_density._vendor.oscbf.manipulator import Manipulator

from xarm_density.core import (
    QuadraticCDFQP,
    ReducedQuadraticCDFQP,
    block_input_map,
    inverse_bump_from_clearance,
    pointwise_transport_row,
    quadratic_basis,
)


DEFAULT_URDF = Path(__file__).resolve().parent / "assets/uf850/arm.urdf"


def make_uf850_collision_model(
    include_gripper: bool = False,
    gripper_opening: float = 0.45,
) -> dict[str, tuple]:
    """Return the same conservative link spheres used by the OSCBF example."""

    wrist_positions = ((0.0, -0.04, 0.0), (0.0, -0.09, 0.0))
    wrist_radii = (0.05, 0.045)
    if include_gripper:
        if gripper_opening >= 0.8:
            # Fixed-closed cover used by the TCP-tracking validation.
            wrist_positions += (
                (0.0, 0.0, 0.035),
                (0.0, 0.0, 0.085),
                (0.0, 0.030, 0.105),
                (0.0, -0.030, 0.105),
                (0.0, 0.024, 0.145),
                (0.0, -0.024, 0.145),
                (0.0, 0.0, 0.165),
            )
            wrist_radii += (0.070, 0.060, 0.045, 0.045, 0.035, 0.035, 0.028)
        else:
            wrist_positions += (
                (0.0, 0.0, 0.035),
                (0.0, 0.0, 0.085),
                (0.0, 0.045, 0.095),
                (0.0, -0.045, 0.095),
                (0.0, 0.050, 0.140),
                (0.0, -0.050, 0.140),
            )
            wrist_radii += (0.070, 0.060, 0.050, 0.050, 0.040, 0.040)

    positions = (
        ((0.0, 0.0, 0.10), (0.0, 0.0, 0.23)),
        ((0.08, 0.0, 0.0), (0.22, 0.0, 0.0), (0.36, 0.0, 0.0)),
        ((0.04, 0.11, 0.0), (0.08, 0.23, 0.0), (0.12, 0.35, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, -0.05, 0.0)),
        ((0.0, 0.0, 0.0),),
        wrist_positions,
    )
    radii = (
        (0.09, 0.09),
        (0.075, 0.075, 0.075),
        (0.07, 0.07, 0.07),
        (0.06, 0.055),
        (0.055,),
        wrist_radii,
    )
    return {"positions": positions, "radii": radii}


@dataclass(frozen=True)
class RobotWorldConfig:
    q_start: tuple[float, ...]
    goal: tuple[float, float, float]
    obstacle_centers: tuple[tuple[float, float, float], ...] = ()
    obstacle_radii: tuple[float, ...] = ()
    safety_mode: str = "none"
    sensing_width: float = 0.12
    alpha: float = 0.2
    v_epsilon: float = 1e-3
    residual_margin: float = 0.01
    qdot_limit: float = 0.7
    nominal_speed: float = 0.22
    nominal_gain: float = 1.4
    dt: float = 0.02
    steps: int = 800
    target_radius: float = 0.035
    ee_radius: float = 0.055
    include_gripper: bool = False
    gripper_opening: float = 0.45
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    urdf: Path = DEFAULT_URDF


class UF850CDFController:
    def __init__(self, config: RobotWorldConfig, qp_backend: str = "reduced"):
        if qp_backend not in {"full", "reduced"}:
            raise ValueError(f"Unknown QP backend: {qp_backend}")
        self.qp_backend = qp_backend
        if config.safety_mode not in {"none", "ee", "whole_body"}:
            raise ValueError(f"Unknown safety mode: {config.safety_mode}")
        self.config = config
        ee_offset = np.eye(4)
        ee_offset[:3, 3] = np.asarray(config.tcp_offset)
        self.robot = Manipulator.from_urdf(
            str(config.urdf),
            ee_offset=ee_offset,
            collision_data=make_uf850_collision_model(
                config.include_gripper, config.gripper_opening
            ),
        )
        centers = jnp.asarray(config.obstacle_centers, dtype=jnp.float64).reshape((-1, 3))
        radii = jnp.asarray(config.obstacle_radii, dtype=jnp.float64)
        goal = jnp.asarray(config.goal, dtype=jnp.float64)
        self.target = goal
        scale = jnp.full(6, jnp.pi)

        def modeled_clearances(q: jnp.ndarray) -> jnp.ndarray:
            if centers.shape[0] == 0 or config.safety_mode == "none":
                return jnp.ones((1,))
            if config.safety_mode == "ee":
                positions = self.robot.ee_position(q)[None, :]
                robot_radii = jnp.asarray([config.ee_radius])
            else:
                collision = self.robot.link_collision_data(q)
                positions = collision[:, :3]
                robot_radii = collision[:, 3]
            delta = positions[:, None, :] - centers[None, :, :]
            distance = jnp.linalg.norm(delta, axis=2)
            return (distance - robot_radii[:, None] - radii[None, :]).reshape(-1)

        def psi(q: jnp.ndarray) -> jnp.ndarray:
            if centers.shape[0] == 0 or config.safety_mode == "none":
                return jnp.asarray(1.0)
            clearance = modeled_clearances(q)
            return jnp.prod(inverse_bump_from_clearance(clearance, config.sensing_width))

        def rho(q: jnp.ndarray, target=goal) -> jnp.ndarray:
            error = self.robot.ee_position(q) - target
            value = jnp.dot(error, error) + config.v_epsilon
            return psi(q) / value**config.alpha

        def flux_basis(q: jnp.ndarray) -> jnp.ndarray:
            return psi(q) * quadratic_basis(q, scale)

        self.ee_position = jax.jit(self.robot.ee_position)
        self.ee_jacobian = jax.jit(jax.jacfwd(self.robot.ee_position))
        self.collision_data = jax.jit(self.robot.link_collision_data)
        self.modeled_clearances = jax.jit(modeled_clearances)
        self.psi = jax.jit(psi)
        self.rho = jax.jit(rho)
        self.flux_basis = jax.jit(flux_basis)
        self.flux_basis_jacobian = jax.jit(jax.jacfwd(flux_basis))
        self.qp_evaluation = jax.jit(lambda q, target: (
            rho(q, target), flux_basis(q), jax.jacfwd(flux_basis)(q),
            psi(q), self.robot.ee_position(q),
        ))
        n_terms = 1 + 6 + 6 * 7 // 2
        qp_type = ReducedQuadraticCDFQP if qp_backend == "reduced" else QuadraticCDFQP
        self.qp = qp_type(
            n_theta=6 * n_terms,
            n_inputs=6,
            residual_margin=config.residual_margin,
            input_bound=config.qdot_limit,
        )

    def nominal_velocity(self, q: np.ndarray) -> np.ndarray:
        position = np.asarray(self.ee_position(jnp.asarray(q)))
        velocity = self.config.nominal_gain * (np.asarray(self.target) - position)
        norm = np.linalg.norm(velocity)
        if norm > self.config.nominal_speed:
            velocity *= self.config.nominal_speed / norm
        return velocity

    def solve(
        self, q: np.ndarray, nominal_qdot: np.ndarray | None = None
    ) -> dict[str, Any]:
        start = perf_counter()
        qj = jnp.asarray(q)
        if self.qp_backend == "reduced":
            rho_j, basis_j, jac_j, psi_j, position_j = self.qp_evaluation(qj, self.target)
            rho = float(rho_j)
            basis = np.asarray(basis_j)
            basis_jacobian = np.asarray(jac_j)
            psi_value, position = float(psi_j), np.asarray(position_j)
        else:
            rho = float(self.rho(qj, self.target))
            basis = np.asarray(self.flux_basis(qj))
            basis_jacobian = np.asarray(self.flux_basis_jacobian(qj))
            psi_value = float(self.psi(qj))
            position = np.asarray(self.ee_position(qj))
        input_map = block_input_map(basis, rho, channels=6)
        if nominal_qdot is None:
            task_map = np.asarray(self.ee_jacobian(qj)) @ input_map
            nominal = self.nominal_velocity(q)
            task_weight = 8.0
        else:
            task_map = input_map
            nominal = np.asarray(nominal_qdot, dtype=float)
            task_weight = 1.0
        result = self.qp.solve(
            input_map=input_map,
            transport_row=pointwise_transport_row(basis_jacobian),
            nominal_input=nominal,
            task_map=task_map,
            task_weight=task_weight,
            input_weight=2e-2,
            theta_weight=2e-4,
        )
        result.update(
            {
                "rho": rho,
                "psi": psi_value,
                "nominal": nominal,
                "ee_position": position,
                "density_floor_active": rho < 1e-10,
            }
        )
        result["controller_time"] = perf_counter() - start
        return result

    def whole_body_clearance(self, q: np.ndarray) -> float:
        if not self.config.obstacle_centers:
            return float("inf")
        collision = np.asarray(self.collision_data(jnp.asarray(q)))
        positions, robot_radii = collision[:, :3], collision[:, 3]
        centers = np.asarray(self.config.obstacle_centers)
        obstacle_radii = np.asarray(self.config.obstacle_radii)
        distance = np.linalg.norm(
            positions[:, None, :] - centers[None, :, :], axis=2
        )
        return float(
            np.min(distance - robot_radii[:, None] - obstacle_radii[None, :])
        )

    def modeled_clearance(self, q: np.ndarray) -> float:
        if not self.config.obstacle_centers:
            return float("inf")
        return float(np.min(np.asarray(self.modeled_clearances(jnp.asarray(q)))))


def simulate_robot_world(config: RobotWorldConfig) -> dict[str, Any]:
    controller = UF850CDFController(config)
    q = np.asarray(config.q_start, dtype=float)
    q_states = [q.copy()]
    ee_states = [np.asarray(controller.ee_position(jnp.asarray(q)))]
    controls = []
    nominal_controls = []
    residuals = []
    modeled_clearances = []
    whole_body_clearances = []
    densities = []
    statuses = []
    solve_times = []
    solver_times = []

    controller.solve(q)
    for _ in range(config.steps):
        if np.linalg.norm(ee_states[-1] - np.asarray(config.goal)) <= config.target_radius:
            break
        result = controller.solve(q)
        qdot = np.asarray(result["input"], dtype=float)
        q = q + config.dt * qdot
        q_states.append(q.copy())
        ee_states.append(np.asarray(controller.ee_position(jnp.asarray(q))))
        controls.append(qdot)
        nominal_controls.append(np.asarray(result["nominal"]))
        residuals.append(float(result["residual"]))
        modeled_clearances.append(controller.modeled_clearance(q))
        whole_body_clearances.append(controller.whole_body_clearance(q))
        densities.append(float(result["rho"]))
        statuses.append(str(result["status"]))
        solve_times.append(float(result["solve_time"]))
        solver_times.append(float(result["solver_solve_time"]))

    q_array = np.asarray(q_states)
    ee_array = np.asarray(ee_states)
    controls_array = np.asarray(controls)
    final_error = float(np.linalg.norm(ee_array[-1] - np.asarray(config.goal)))
    whole_array = np.asarray(whole_body_clearances)
    modeled_array = np.asarray(modeled_clearances)
    return {
        "q_states": q_array,
        "ee_states": ee_array,
        "controls": controls_array,
        "nominal_controls": np.asarray(nominal_controls),
        "residuals": np.asarray(residuals),
        "modeled_clearances": modeled_array,
        "whole_body_clearances": whole_array,
        "densities": np.asarray(densities),
        "statuses": statuses,
        "solve_times": np.asarray(solve_times),
        "solver_times": np.asarray(solver_times),
        "start_collision_data": np.asarray(controller.collision_data(jnp.asarray(q_array[0]))),
        "final_collision_data": np.asarray(controller.collision_data(jnp.asarray(q_array[-1]))),
        "metrics": {
            "steps": int(controls_array.shape[0]),
            "target_reached": bool(final_error <= config.target_radius),
            "final_target_error": final_error,
            "minimum_modeled_clearance": (
                float(np.min(modeled_array))
                if config.obstacle_centers and modeled_array.size
                else None
            ),
            "minimum_whole_body_clearance": (
                float(np.min(whole_array))
                if config.obstacle_centers and whole_array.size
                else None
            ),
            "minimum_cdf_residual": (
                float(np.min(residuals)) if residuals else None
            ),
            "maximum_joint_speed": (
                float(np.max(np.abs(controls_array))) if controls_array.size else 0.0
            ),
            "all_qps_feasible": bool(
                all(status.startswith("optimal") for status in statuses)
            ),
            "mean_solver_time_ms": (
                float(1e3 * np.mean(solver_times)) if solver_times else 0.0
            ),
            "mean_controller_qp_time_ms": (
                float(1e3 * np.mean(solve_times)) if solve_times else 0.0
            ),
        },
    }
