"""OSCBF-compatible UF850 PyBullet validation harness for the CDF-QP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from oscbf.core.controllers import PoseTaskVelocityController

from xarm_cdf_qp.robot import RobotWorldConfig, UF850CDFController


RESULTS_ROOT = Path("/home/sriramk/xarm_ws/results/cdf_qp_validations")


@dataclass(frozen=True)
class PyBulletValidationConfig:
    index: int
    name: str
    purpose: str
    filter_mode: str
    safety_mode: str
    q_start: tuple[float, ...] = (0.0, -0.7, -1.0, 0.0, 0.8, 0.0)
    goal: tuple[float, float, float] = (0.40, 0.35, 0.45)
    obstacle_centers: tuple[tuple[float, float, float], ...] = ()
    obstacle_radii: tuple[float, ...] = ()
    dt: float = 1.0 / 240.0
    steps: int = 1200
    target_radius: float = 0.035
    qdot_limit: float = 0.7
    kp_task: float = 1.4
    kp_posture: float = 0.2
    sensing_width: float = 0.16
    residual_margin: float = 0.01
    capture_every: int = 12
    gif_fps: int = 20
    with_gripper: bool = False
    gripper_opening: float = 0.45
    track_tcp: bool = False
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    track_orientation: bool = False
    kp_rotation: float = 0.0
    reference_duration_s: float = 0.0


def validation_config(index: int) -> PyBulletValidationConfig:
    common = {
        "q_start": (0.0, -0.7, -1.0, 0.0, 0.8, 0.0),
        "goal": (0.40, 0.35, 0.45),
    }
    if index == 1:
        return PyBulletValidationConfig(
            index=1,
            name="Nominal OSCBF harness baseline",
            purpose="Verify the unchanged OSCBF nominal controller and PyBullet playback.",
            filter_mode="nominal",
            safety_mode="none",
            **common,
        )
    if index == 2:
        return PyBulletValidationConfig(
            index=2,
            name="Obstacle-free CDF-QP",
            purpose="Check that the CDF-QP tracks the OSCBF nominal command without obstacles.",
            filter_mode="cdf",
            safety_mode="none",
            **common,
        )
    if index == 3:
        return PyBulletValidationConfig(
            index=3,
            name="Single-obstacle whole-arm CDF-QP",
            purpose="Avoid one sphere on the nominal workspace path using all link spheres.",
            filter_mode="cdf",
            safety_mode="whole_body",
            obstacle_centers=((0.50, 0.18, 0.36),),
            obstacle_radii=(0.065,),
            **common,
        )
    if index == 4:
        return PyBulletValidationConfig(
            index=4,
            name="Multiple-obstacle whole-arm CDF-QP",
            purpose="Test the product density window with deterministic spherical obstacles.",
            filter_mode="cdf",
            safety_mode="whole_body",
            obstacle_centers=((0.50, 0.18, 0.36), (0.45, 0.35, 0.30)),
            obstacle_radii=(0.055, 0.040),
            **common,
        )
    if index == 5:
        return PyBulletValidationConfig(
            index=5,
            name="End-effector-only density counterexample",
            purpose="Show that EE clearance does not imply clearance for intermediate links.",
            filter_mode="cdf",
            safety_mode="ee",
            obstacle_centers=((0.25, 0.15, 0.40),),
            obstacle_radii=(0.085,),
            **common,
        )
    if index == 6:
        return PyBulletValidationConfig(
            index=6,
            name="Whole-arm density",
            purpose="Use the validation-5 scene while enforcing density safety for every link sphere.",
            filter_mode="cdf",
            safety_mode="whole_body",
            obstacle_centers=((0.25, 0.15, 0.40),),
            obstacle_radii=(0.085,),
            **common,
        )
    if index == 7:
        return PyBulletValidationConfig(
            index=7,
            name="Cluttered multi-sphere whole-arm CDF-QP",
            purpose=(
                "Find a collision-free whole-arm path through a deterministic "
                "cluttered workspace containing several spherical obstacles."
            ),
            filter_mode="cdf",
            safety_mode="whole_body",
            obstacle_centers=(
                (0.25, 0.15, 0.40),
                (0.58, 0.314, 0.42),
                (0.12, 0.314, 0.54),
                (0.455, 0.44, 0.62),
            ),
            obstacle_radii=(0.085, 0.035, 0.035, 0.035),
            steps=3600,
            **common,
        )
    if index == 8:
        return PyBulletValidationConfig(
            index=8,
            name="Gripper-equipped cluttered whole-arm CDF-QP",
            purpose=(
                "Validate a collision-free path through a cluttered workspace "
                "with a fixed-open xArm gripper and gripper safety spheres."
            ),
            filter_mode="cdf",
            safety_mode="whole_body",
            obstacle_centers=(
                (0.25, 0.15, 0.40),
                (0.58, 0.314, 0.42),
                (0.12, 0.314, 0.54),
            ),
            obstacle_radii=(0.040, 0.035, 0.035),
            steps=3600,
            with_gripper=True,
            **common,
        )
    if index == 9:
        return PyBulletValidationConfig(
            index=9,
            name="Closed-gripper TCP trajectory tracking in clutter",
            purpose=(
                "Track a smooth Cartesian reference with the closed-gripper "
                "fingertip midpoint while the whole-arm CDF-QP avoids a dense "
                "field of larger spherical obstacles."
            ),
            filter_mode="cdf",
            safety_mode="whole_body",
            goal=(0.38, 0.35, 0.43),
            obstacle_centers=(
                (0.5665, 0.2553, 0.3056),
                (0.4900, 0.2844, 0.5415),
                (0.2924, 0.2063, 0.2649),
                (0.6790, 0.0892, 0.4238),
                (0.5021, 0.3967, 0.3615),
            ),
            obstacle_radii=(0.075, 0.080, 0.076, 0.084, 0.061),
            sensing_width=0.10,
            steps=4800,
            with_gripper=True,
            gripper_opening=0.85,
            track_tcp=True,
            tcp_offset=(0.0, 0.0, 0.172),
            track_orientation=True,
            kp_rotation=1.0,
            reference_duration_s=7.0,
            q_start=common["q_start"],
        )
    raise ValueError("Validation index must be in 1..9")


def _setup_bullet(config: PyBulletValidationConfig):
    import pybullet as p
    import pybullet_data

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(config.dt)
    p.setGravity(0.0, 0.0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF("plane.urdf")
    urdf = (
        "/home/sriramk/xarm_ws/generated/uf850_oscbf_pybullet_gripper.urdf"
        if config.with_gripper
        else "/home/sriramk/xarm_ws/generated/uf850_oscbf_pybullet.urdf"
    )
    robot_id = p.loadURDF(urdf, useFixedBase=True)
    joint_indices = []
    gripper_joint_indices = []
    tcp_link_index = None
    for joint in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint)
        name = info[1].decode()
        link_name = info[12].decode()
        if link_name == "link_tcp":
            tcp_link_index = joint
        if name in {f"joint{i}" for i in range(1, 7)}:
            joint_indices.append(joint)
        elif config.with_gripper and info[2] != p.JOINT_FIXED:
            gripper_joint_indices.append(joint)
    joint_indices.sort()
    for joint, value in zip(joint_indices, config.q_start):
        p.resetJointState(robot_id, joint, value)
    for joint in gripper_joint_indices:
        p.resetJointState(robot_id, joint, config.gripper_opening)

    obstacle_ids = []
    for center, radius in zip(config.obstacle_centers, config.obstacle_radii):
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=(0.35, 0.35, 0.35, 1.0)
        )
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        obstacle_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=center,
            )
        )
    target_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.025, rgbaColor=(0.10, 0.65, 0.25, 1.0)
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=target_visual,
        basePosition=config.goal,
    )
    p.stepSimulation()
    tracked_link_index = (
        tcp_link_index
        if config.track_tcp and tcp_link_index is not None
        else joint_indices[-1]
    )
    return (
        p,
        client,
        robot_id,
        joint_indices,
        gripper_joint_indices,
        tracked_link_index,
        obstacle_ids,
    )


def _bullet_clearance(p, robot_id: int, obstacle_ids: list[int]) -> tuple[float, int]:
    if not obstacle_ids:
        return float("inf"), 0
    minimum = float("inf")
    contacts = 0
    for obstacle_id in obstacle_ids:
        points = p.getClosestPoints(robot_id, obstacle_id, distance=2.0)
        if points:
            minimum = min(minimum, min(float(point[8]) for point in points))
        contacts += len(p.getContactPoints(robot_id, obstacle_id))
    return minimum, contacts


def _render_frame(p) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(0.30, 0.15, 0.38),
        distance=1.25,
        yaw=45.0,
        pitch=-24.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=48.0, aspect=4.0 / 3.0, nearVal=0.05, farVal=3.0
    )
    _, _, rgba, _, _ = p.getCameraImage(
        640,
        480,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=(1.0, -1.0, 2.0),
    )
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8)[:, :, :3])


def run_pybullet_validation(
    index: int, output_root: Path = RESULTS_ROOT
) -> dict[str, Any]:
    config = validation_config(index)
    output_dir = output_root / f"validation_{index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_config = RobotWorldConfig(
        q_start=config.q_start,
        goal=config.goal,
        obstacle_centers=config.obstacle_centers,
        obstacle_radii=config.obstacle_radii,
        safety_mode=config.safety_mode,
        sensing_width=config.sensing_width,
        residual_margin=config.residual_margin,
        qdot_limit=config.qdot_limit,
        dt=config.dt,
        steps=config.steps,
        target_radius=config.target_radius,
        include_gripper=config.with_gripper,
        gripper_opening=config.gripper_opening,
        tcp_offset=config.tcp_offset,
    )
    cdf = UF850CDFController(controller_config)
    robot = cdf.robot
    nominal_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.asarray([config.kp_task] * 3 + [config.kp_rotation] * 3),
        kp_joint=config.kp_posture,
        qdot_min=-config.qdot_limit * np.ones(robot.num_joints),
        qdot_max=config.qdot_limit * np.ones(robot.num_joints),
    )
    goal = jnp.asarray(config.goal)
    desired_q = jnp.asarray(config.q_start)
    initial_transform = np.asarray(
        robot.ee_transform(jnp.asarray(config.q_start))
    )
    initial_position = initial_transform[:3, 3]
    desired_rotation = jnp.asarray(initial_transform[:3, :3])
    zero3 = jnp.zeros(3)

    @jax.jit
    def nominal_control(q, desired_position, desired_velocity):
        m_inv, jacobian, ee_transform = robot.dynamically_consistent_velocity_control_matrices(q)
        return nominal_controller(
            q,
            ee_transform[:3, 3],
            ee_transform[:3, :3],
            desired_position,
            desired_rotation,
            desired_velocity,
            zero3,
            desired_q,
            jacobian,
            m_inv,
        )

    def reference_at(time_s: float) -> tuple[np.ndarray, np.ndarray]:
        duration = config.reference_duration_s
        if duration <= 0.0:
            return np.asarray(config.goal), np.zeros(3)
        phase = np.clip(time_s / duration, 0.0, 1.0)
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        blend_rate = (
            30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4
        ) / duration
        displacement = np.asarray(config.goal) - initial_position
        return initial_position + blend * displacement, blend_rate * displacement

    (
        p,
        client,
        robot_id,
        joint_indices,
        gripper_joint_indices,
        tracked_link_index,
        obstacle_ids,
    ) = _setup_bullet(config)
    q = np.asarray(config.q_start, dtype=float)
    ee_jax = np.asarray(cdf.ee_position(jnp.asarray(q)))
    trail_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.006, rgbaColor=(0.05, 0.35, 0.90, 1.0)
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=trail_visual,
        basePosition=ee_jax,
    )
    frames = [_render_frame(p)]
    q_states = [q.copy()]
    ee_jax_states = [ee_jax.copy()]
    ee_bullet_states = [
        np.asarray(p.getLinkState(robot_id, tracked_link_index, 1)[4])
    ]
    reference_positions = [initial_position.copy()]
    orientation_errors = [0.0]
    nominal_controls = []
    filtered_controls = []
    residuals = []
    modeled_clearances = []
    sphere_clearances = []
    bullet_clearances = []
    contact_counts = []
    statuses = []
    solve_times = []
    solver_times = []

    warm_position, warm_velocity = reference_at(0.0)
    nominal_control(
        jnp.asarray(q), jnp.asarray(warm_position), jnp.asarray(warm_velocity)
    ).block_until_ready()
    if config.filter_mode == "cdf":
        warm_nominal = np.asarray(
            nominal_control(
                jnp.asarray(q), jnp.asarray(warm_position), jnp.asarray(warm_velocity)
            )
        )
        cdf.solve(q, warm_nominal)

    for step in range(config.steps):
        reference_position, reference_velocity = reference_at(step * config.dt)
        nominal = np.asarray(
            nominal_control(
                jnp.asarray(q),
                jnp.asarray(reference_position),
                jnp.asarray(reference_velocity),
            )
        )
        if config.filter_mode == "cdf":
            solution = cdf.solve(q, nominal)
            qdot = np.asarray(solution["input"])
            residual = float(solution["residual"])
            status = str(solution["status"])
            solve_time = float(solution["solve_time"])
            solver_time = float(solution["solver_solve_time"])
        else:
            qdot = nominal
            residual = float("nan")
            status = "nominal"
            solve_time = 0.0
            solver_time = 0.0

        q = q + config.dt * qdot
        for joint, value in zip(joint_indices, q):
            p.resetJointState(robot_id, joint, float(value))
        for joint in gripper_joint_indices:
            p.resetJointState(robot_id, joint, config.gripper_opening)
        p.stepSimulation()
        ee_jax = np.asarray(cdf.ee_position(jnp.asarray(q)))
        ee_bullet = np.asarray(p.getLinkState(robot_id, tracked_link_index, 1)[4])
        next_reference, _ = reference_at((step + 1) * config.dt)
        rotation = np.asarray(robot.ee_transform(jnp.asarray(q)))[:3, :3]
        rotation_cosine = np.clip(
            0.5 * (np.trace(initial_transform[:3, :3].T @ rotation) - 1.0),
            -1.0,
            1.0,
        )
        physical_clearance, contacts = _bullet_clearance(p, robot_id, obstacle_ids)
        if contacts:
            for obstacle_id in obstacle_ids:
                p.changeVisualShape(
                    obstacle_id, -1, rgbaColor=(0.80, 0.12, 0.12, 1.0)
                )
        sphere_clearance = cdf.whole_body_clearance(q)
        modeled_clearance = cdf.modeled_clearance(q)

        q_states.append(q.copy())
        ee_jax_states.append(ee_jax.copy())
        ee_bullet_states.append(ee_bullet.copy())
        reference_positions.append(next_reference.copy())
        orientation_errors.append(float(np.arccos(rotation_cosine)))
        nominal_controls.append(nominal)
        filtered_controls.append(qdot)
        residuals.append(residual)
        modeled_clearances.append(modeled_clearance)
        sphere_clearances.append(sphere_clearance)
        bullet_clearances.append(physical_clearance)
        contact_counts.append(contacts)
        statuses.append(status)
        solve_times.append(solve_time)
        solver_times.append(solver_time)

        if (step + 1) % config.capture_every == 0:
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=trail_visual,
                basePosition=ee_jax,
            )
            frames.append(_render_frame(p))
        reference_finished = (
            config.reference_duration_s <= 0.0
            or (step + 1) * config.dt >= config.reference_duration_s
        )
        if reference_finished and np.linalg.norm(
            ee_jax - np.asarray(config.goal)
        ) <= config.target_radius:
            break

    if (len(q_states) - 1) % config.capture_every:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=trail_visual,
            basePosition=ee_jax,
        )
        frames.append(_render_frame(p))
    p.disconnect(client)

    arrays = {
        "q_states": np.asarray(q_states),
        "ee_jax_states": np.asarray(ee_jax_states),
        "ee_bullet_states": np.asarray(ee_bullet_states),
        "reference_positions": np.asarray(reference_positions),
        "orientation_errors": np.asarray(orientation_errors),
        "nominal_controls": np.asarray(nominal_controls),
        "filtered_controls": np.asarray(filtered_controls),
        "residuals": np.asarray(residuals),
        "modeled_clearances": np.asarray(modeled_clearances),
        "sphere_clearances": np.asarray(sphere_clearances),
        "bullet_clearances": np.asarray(bullet_clearances),
        "contact_counts": np.asarray(contact_counts),
        "solve_times": np.asarray(solve_times),
        "solver_times": np.asarray(solver_times),
    }
    metrics = _metrics(config, arrays, statuses)
    _save_artifacts(config, arrays, metrics, frames, output_dir)
    return {"config": config, "metrics": metrics, **arrays}


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else None


def _metrics(
    config: PyBulletValidationConfig,
    arrays: dict[str, np.ndarray],
    statuses: list[str],
) -> dict[str, Any]:
    error = np.linalg.norm(arrays["ee_jax_states"][-1] - np.asarray(config.goal))
    fk_error = np.linalg.norm(
        arrays["ee_jax_states"] - arrays["ee_bullet_states"], axis=1
    )
    reference_error = np.linalg.norm(
        arrays["ee_jax_states"] - arrays["reference_positions"], axis=1
    )
    residual = arrays["residuals"]
    finite_residual = residual[np.isfinite(residual)]
    return {
        "steps": int(arrays["filtered_controls"].shape[0]),
        "simulated_time_s": float(config.dt * arrays["filtered_controls"].shape[0]),
        "target_reached": bool(error <= config.target_radius),
        "final_target_error_m": float(error),
        "maximum_reference_tracking_error_m": float(np.max(reference_error)),
        "final_orientation_error_deg": float(
            np.rad2deg(arrays["orientation_errors"][-1])
        ),
        "maximum_orientation_error_deg": float(
            np.rad2deg(np.max(arrays["orientation_errors"]))
        ),
        "minimum_modeled_clearance_m": _finite_min(arrays["modeled_clearances"]),
        "minimum_link_sphere_clearance_m": _finite_min(arrays["sphere_clearances"]),
        "minimum_pybullet_mesh_clearance_m": _finite_min(arrays["bullet_clearances"]),
        "pybullet_contact_steps": int(np.count_nonzero(arrays["contact_counts"])),
        "minimum_cdf_residual": (
            float(np.min(finite_residual)) if finite_residual.size else None
        ),
        "all_qps_feasible": bool(
            config.filter_mode == "nominal"
            or all(status.startswith("optimal") for status in statuses)
        ),
        "maximum_joint_speed_rad_s": float(
            np.max(np.abs(arrays["filtered_controls"]))
        ),
        "maximum_jax_bullet_fk_error_m": float(np.max(fk_error)),
        "mean_qp_time_ms": float(1e3 * np.mean(arrays["solve_times"])),
        "maximum_qp_time_ms": float(1e3 * np.max(arrays["solve_times"])),
        "mean_solver_time_ms": float(1e3 * np.mean(arrays["solver_times"])),
    }


def _save_artifacts(
    config: PyBulletValidationConfig,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
    frames: list[Image.Image],
    output_dir: Path,
) -> None:
    np.savez_compressed(output_dir / "trajectory.npz", **arrays)
    payload = {"config": asdict(config), "metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    frames[0].save(
        output_dir / "pybullet.gif",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / config.gif_fps),
        loop=0,
        optimize=True,
    )
    _save_plot(config, arrays, output_dir / "result.png")
    _save_markdown(config, metrics, output_dir)


def _save_plot(
    config: PyBulletValidationConfig,
    arrays: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    time = np.arange(arrays["filtered_controls"].shape[0]) * config.dt
    state_time = np.arange(arrays["ee_jax_states"].shape[0]) * config.dt
    error = np.linalg.norm(
        arrays["ee_jax_states"] - np.asarray(config.goal), axis=1
    )
    reference_error = np.linalg.norm(
        arrays["ee_jax_states"] - arrays["reference_positions"], axis=1
    )
    fk_error = np.linalg.norm(
        arrays["ee_jax_states"] - arrays["ee_bullet_states"], axis=1
    )
    fig = plt.figure(figsize=(10.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    trajectory = fig.add_subplot(grid[0, 0], projection="3d")
    diagnostics = [fig.add_subplot(grid[row, col]) for row, col in ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1))]
    ee = arrays["ee_jax_states"]
    trajectory.plot(*ee.T, color="#0072B2", lw=2.0)
    if config.reference_duration_s > 0.0:
        trajectory.plot(
            *arrays["reference_positions"].T,
            color="0.35",
            ls="--",
            lw=1.1,
            label="reference",
        )
    trajectory.scatter(*ee[0], color="black", s=24, label="start")
    trajectory.scatter(*config.goal, marker="*", color="#009E73", s=80, label="target")
    for center, radius in zip(config.obstacle_centers, config.obstacle_radii):
        trajectory.scatter(*center, color="0.35", s=900 * radius, alpha=0.8)
    trajectory.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=config.name)
    trajectory.legend(frameon=True, fontsize=8)

    diagnostics[0].plot(state_time, error, color="#0072B2", label="target")
    if config.reference_duration_s > 0.0:
        diagnostics[0].plot(
            state_time, reference_error, color="#D55E00", label="reference"
        )
        diagnostics[0].legend(frameon=True, fontsize=8)
    diagnostics[0].axhline(config.target_radius, color="0.4", ls="--", lw=0.8)
    diagnostics[0].set_ylabel("Target error [m]")
    diagnostics[1].plot(time, arrays["filtered_controls"], lw=0.9)
    diagnostics[1].axhline(config.qdot_limit, color="0.35", ls="--", lw=0.8)
    diagnostics[1].axhline(-config.qdot_limit, color="0.35", ls="--", lw=0.8)
    diagnostics[1].set_ylabel("Filtered $\\dot q$ [rad/s]")
    residual = arrays["residuals"]
    if np.any(np.isfinite(residual)):
        diagnostics[2].plot(time, residual, color="#D55E00")
        diagnostics[2].axhline(0.0, color="0.35", ls="--", lw=0.8)
    else:
        diagnostics[2].text(0.5, 0.5, "Nominal baseline", ha="center", va="center", transform=diagnostics[2].transAxes)
    diagnostics[2].set_ylabel("CDF residual")
    for values, label, color in (
        (arrays["modeled_clearances"], "modeled", "#009E73"),
        (arrays["sphere_clearances"], "all spheres", "#CC79A7"),
        (arrays["bullet_clearances"], "Bullet mesh", "#E69F00"),
    ):
        if np.any(np.isfinite(values)):
            diagnostics[3].plot(time, values, label=label, color=color)
    diagnostics[3].axhline(0.0, color="0.35", ls="--", lw=0.8)
    diagnostics[3].set_ylabel("Clearance [m]")
    handles, labels = diagnostics[3].get_legend_handles_labels()
    if labels:
        diagnostics[3].legend(frameon=True, fontsize=8)
    elif not config.obstacle_centers:
        diagnostics[3].text(
            0.5, 0.5, "No obstacles", ha="center", va="center",
            transform=diagnostics[3].transAxes,
        )
    if config.track_orientation:
        diagnostics[4].plot(
            state_time,
            np.rad2deg(arrays["orientation_errors"]),
            color="#0072B2",
        )
        diagnostics[4].set_ylabel("Orientation error [deg]")
    else:
        diagnostics[4].semilogy(
            state_time, np.maximum(fk_error, 1e-12), color="#0072B2"
        )
        diagnostics[4].set_ylabel("JAX/Bullet FK error [m]")
    for axis in diagnostics:
        axis.set_xlabel("Time [s]")
        axis.grid(color="0.88", lw=0.7)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_markdown(
    config: PyBulletValidationConfig,
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    result_rows = "\n".join(
        f"| `{key}` | `{value}` |" for key, value in metrics.items()
    )
    parameters = (
        f"| Timestep | `{config.dt:.8f} s` |\n"
        f"| Filter | `{config.filter_mode}` |\n"
        f"| Safety geometry | `{config.safety_mode}` |\n"
        f"| Joint-rate limit | `{config.qdot_limit} rad/s` |\n"
        f"| Obstacles | `{len(config.obstacle_centers)}` |"
        f"\n| Gripper enabled | `{config.with_gripper}` |"
        f"\n| Gripper opening | `{config.gripper_opening} rad` |"
        f"\n| Tracked frame | `{'link_tcp' if config.track_tcp else 'link6'}` |"
        f"\n| TCP offset | `{config.tcp_offset} m` |"
        f"\n| Reference duration | `{config.reference_duration_s} s` |"
        f"\n| Orientation gain | `{config.kp_rotation}` |"
    )
    findings = _finding(config, metrics)
    readme = f"""# Validation {config.index}: {config.name}

{config.purpose}

![PyBullet playback](pybullet.gif)

![Validation diagnostics](result.png)

## Setup

| Parameter | Value |
|---|---:|
{parameters}

The runner matches the OSCBF playground: the pose-task controller produces
`u_nom`, the selected filter returns `qdot`, and the state is advanced with
`q_next = q + dt*qdot` before `resetJointState` and one `stepSimulation` call.

## Finding

{findings}

Detailed metrics are listed in [results.md](results.md).
"""
    results = f"""# Validation {config.index} results

## Goal

{config.purpose}

## Metrics

| Metric | Value |
|---|---:|
{result_rows}

## Interpretation

{findings}

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
"""
    (output_dir / "README.md").write_text(readme)
    (output_dir / "results.md").write_text(results)


def _finding(config: PyBulletValidationConfig, metrics: dict[str, Any]) -> str:
    reached = "reached" if metrics["target_reached"] else "did not reach"
    contact = metrics["pybullet_contact_steps"]
    clearance = metrics["minimum_link_sphere_clearance_m"]
    if config.index == 5:
        return (
            f"The robot {reached} the target. The end-effector-only filter left "
            f"the remaining links unmodeled; the minimum all-sphere clearance was "
            f"`{clearance}` m and Bullet reported contact on `{contact}` steps."
        )
    if config.index == 6:
        return (
            f"The robot {reached} the target with all link spheres included. "
            f"The minimum all-sphere clearance was `{clearance}` m and Bullet "
            f"reported contact on `{contact}` steps."
        )
    if config.index == 7:
        return (
            f"The whole-arm CDF-QP {reached} the target through a cluttered "
            f"field of `{len(config.obstacle_centers)}` spheres. The minimum "
            f"all-sphere clearance was `{clearance}` m and Bullet reported "
            f"contact on `{contact}` steps."
        )
    if config.index == 8:
        return (
            f"The gripper-equipped whole-arm CDF-QP {reached} the target "
            f"through a `{len(config.obstacle_centers)}`-sphere clutter field "
            f"sized for the larger gripper footprint. The minimum arm-and-"
            f"gripper sphere clearance was `{clearance}` m and Bullet reported "
            f"contact on `{contact}` steps."
        )
    if config.index == 9:
        return (
            f"The closed-gripper TCP CDF-QP {reached} the final pose after "
            f"tracking a minimum-jerk Cartesian reference through a "
            f"`{len(config.obstacle_centers)}`-sphere clutter field. The "
            f"minimum arm-and-gripper sphere clearance was `{clearance}` m, "
            f"the final orientation error was "
            f"`{metrics['final_orientation_error_deg']}` deg, and Bullet "
            f"reported contact on `{contact}` steps."
        )
    return (
        f"The robot {reached} the target. Bullet reported contact on `{contact}` "
        f"steps, and all configured QPs were feasible: "
        f"`{metrics['all_qps_feasible']}`."
    )


def write_parent_results(
    summaries: dict[int, dict[str, Any]], output_root: Path = RESULTS_ROOT
) -> None:
    rows = []
    sections = []
    for index, metrics in summaries.items():
        config = validation_config(index)
        rows.append(
            f"| [{index}](validation_{index:02d}/README.md) | {config.name} | "
            f"{metrics['target_reached']} | {metrics['minimum_link_sphere_clearance_m']} | "
            f"{metrics['minimum_pybullet_mesh_clearance_m']} | {metrics['pybullet_contact_steps']} |"
        )
        sections.append(
            f"## Validation {index}\n\n"
            f"![Validation {index} GIF](validation_{index:02d}/pybullet.gif)\n\n"
            f"![Validation {index} plot](validation_{index:02d}/result.png)\n\n"
            f"[Read the validation summary](validation_{index:02d}/README.md) | "
            f"[Detailed results](validation_{index:02d}/results.md)"
        )
    text = """# UF850 CDF-QP PyBullet validations

These tests reuse the OSCBF UF850 PyBullet harness and nominal pose-task
controller. Only the safety-filter optimization is changed to the pointwise
CDF-QP where indicated. PyBullet is used for kinematic playback, rendering,
mesh clearance/contact checks, and an independent forward-kinematics check.

| ID | Validation | Reached | Min sphere clearance [m] | Min Bullet clearance [m] | Contact steps |
|---:|---|:---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\n" + "\n\n".join(sections) + """

## Gripper status and remaining TODO

- Completed in validation 8: load the gripper-enabled UF850 URDF, hold all six
  gripper joints at a fixed `0.45 rad` opening, add six conservative palm and
  finger spheres to the JAX model, and check both sphere and Bullet mesh
  clearance in a three-obstacle cluttered scene.
- Completed in validation 9: track the URDF-defined `link_tcp` at a `0.172 m`
  wrist offset with the gripper closed at `0.85 rad`, preserve a fixed tool
  orientation, and filter a smooth Cartesian reference through denser clutter.
- Generalize the collision spheres to follow a commanded gripper opening
  instead of assuming one fixed configuration.
- Validate the gripper sphere cover against Bullet mesh distances over the
  full opening range.
- Include at least one narrow-passage case where safety depends on coordinated
  clearance of the upper arm, wrist, and gripper rather than only the end
  effector.
- Compare CDF-QP and OSCBF using identical starts, goals, obstacle fields,
  nominal controls, rate limits, timestep, and collision geometry.
- Report convergence, modeled and Bullet clearance, contacts, residuals,
  feasibility, input limits, solve time, plots, and PyBullet GIFs for every
  cluttered scene.
"""
    (output_root / "results.md").write_text(text)
    (output_root / "README.md").write_text(text)
