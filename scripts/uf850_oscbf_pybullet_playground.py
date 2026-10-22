#!/usr/bin/env python3
"""ROS-free UF850 OSCBF PyBullet playground.

This is intentionally a lightweight sandbox:

- generate a UF850 URDF from the local xarm_ros2 xacro if needed;
- load UF850 kinematics into OSCBF's Manipulator class;
- load a UF850 URDF in PyBullet for visualization, optionally with gripper;
- use a nominal operational-space velocity controller;
- filter it through an OSCBF velocity safety filter;
- integrate q <- q + dt * qdot_safe.

It is not a MoveIt or ROS controller. It is a fast place to play with starts,
goals, obstacle counts, and collision-sphere models before we touch ROS again.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


XARM_WS = Path("/home/sriramk/xarm_ws")
XARM_ROS2 = XARM_WS / "src" / "xarm_ros2"
XARM_SETUP = XARM_WS / "install" / "setup.bash"
DEFAULT_URDF = XARM_WS / "generated" / "uf850_oscbf_pybullet.urdf"
DEFAULT_GRIPPER_URDF = XARM_WS / "generated" / "uf850_oscbf_pybullet_gripper.urdf"
XACRO_FILE = XARM_ROS2 / "xarm_description" / "urdf" / "xarm_device.urdf.xacro"


@dataclass(frozen=True)
class PlaygroundConfig:
    dt: float
    kp_task: float
    kp_posture: float
    alpha: float
    qdot_limit: float
    safety_margin: float


def require_dependencies() -> None:
    missing = []
    for module_name in ("jax", "jax.numpy", "cbfpy", "oscbf", "pybullet"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {names}\n\n"
            "Activate the xarm conda env first:\n"
            "  source /home/sriramk/anaconda3/etc/profile.d/conda.sh\n"
            "  conda activate xarm\n"
        )


def generate_uf850_urdf(output_path: Path, force: bool = False, add_gripper: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not XARM_SETUP.exists():
        raise FileNotFoundError(f"Missing ROS setup for xacro generation: {XARM_SETUP}")
    if not XACRO_FILE.exists():
        raise FileNotFoundError(f"Missing xArm xacro: {XACRO_FILE}")
    command = (
        f"source {shlex.quote(str(XARM_SETUP))} && "
        f"xacro {shlex.quote(str(XACRO_FILE))} "
        f"dof:=6 robot_type:=uf850 add_gripper:={'true' if add_gripper else 'false'} "
        "add_bio_gripper:=false "
        "add_vacuum_gripper:=false add_realsense_d435i:=false "
        "limited:=true load_gazebo_plugin:=false "
        f"-o {shlex.quote(str(output_path))}"
    )
    subprocess.run(["bash", "-lc", command], check=True)
    # OSCBF's bundled urdfpy parser does not resolve ROS package:// URLs.
    # Rewrite xarm_description mesh paths to absolute paths for ROS-free use.
    description_root = XARM_ROS2 / "xarm_description"
    urdf_text = output_path.read_text()
    urdf_text = urdf_text.replace(
        "package://xarm_description/",
        str(description_root) + "/",
    )
    output_path.write_text(urdf_text)
    return output_path


def make_uf850_collision_model(radius_scale: float, margin: float) -> dict[str, tuple]:
    """Manual OSCBF sphere model, one tuple entry per controlled link.

    The positions are rough local-frame sphere centers. They are intentionally
    conservative enough for early experiments, but not a final collision model.
    """
    r = lambda value: float(radius_scale * value + margin)
    positions = (
        ((0.0, 0.0, 0.10), (0.0, 0.0, 0.23)),
        ((0.08, 0.0, 0.0), (0.22, 0.0, 0.0), (0.36, 0.0, 0.0)),
        ((0.04, 0.11, 0.0), (0.08, 0.23, 0.0), (0.12, 0.35, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, -0.05, 0.0)),
        ((0.0, 0.0, 0.0),),
        ((0.0, -0.04, 0.0), (0.0, -0.09, 0.0)),
    )
    radii = (
        (r(0.09), r(0.09)),
        (r(0.075), r(0.075), r(0.075)),
        (r(0.07), r(0.07), r(0.07)),
        (r(0.06), r(0.055)),
        (r(0.055),),
        (r(0.05), r(0.045)),
    )
    return {"positions": positions, "radii": radii}


def sample_obstacles(num_obs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if num_obs <= 0:
        return np.zeros((0, 3)), np.zeros((0,))
    positions = rng.uniform(low=[0.25, -0.35, 0.12], high=[0.85, 0.35, 0.72], size=(num_obs, 3))
    radii = rng.uniform(0.035, 0.09, size=(num_obs,))
    return positions, radii


def parse_vec(text: str, expected: int, name: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.shape != (expected,):
        raise argparse.ArgumentTypeError(f"{name} must have {expected} comma-separated values")
    return values


def make_velocity_config_class():
    import jax
    import jax.numpy as jnp
    from oscbf.core.oscbf_configs import OSCBFVelocityConfig

    @jax.tree_util.register_static
    class UF850CollisionVelocityConfig(OSCBFVelocityConfig):
        def __init__(self, robot, obstacle_positions, obstacle_radii, z_min, alpha_gain):
            self.obstacle_positions = np.atleast_2d(obstacle_positions)
            self.obstacle_radii = np.ravel(obstacle_radii)
            self.z_min = float(z_min)
            self.alpha_gain = float(alpha_gain)
            super().__init__(robot)

        def h_1(self, z, **kwargs):
            q = z[: self.num_joints]
            robot_collision = self.robot.link_collision_data(q)
            robot_pos = robot_collision[:, :3]
            robot_rad = robot_collision[:, 3, None]
            if self.obstacle_positions.size:
                deltas = (robot_pos[:, None, :] - self.obstacle_positions[None, :, :]).reshape(-1, 3)
                radii = (robot_rad[:, None] + self.obstacle_radii[None, :]).reshape(-1)
                h_obs = jnp.linalg.norm(deltas, axis=1) - radii
            else:
                h_obs = jnp.zeros((0,))
            h_table = robot_pos[:, 2] - self.z_min - robot_collision[:, 3]
            return jnp.concatenate([h_obs, h_table])

        def alpha(self, h):
            return self.alpha_gain * h

    return UF850CollisionVelocityConfig


def setup_pybullet(urdf_path: Path, q0: np.ndarray, obstacles: tuple[np.ndarray, np.ndarray], gui: bool):
    import pybullet as p
    import pybullet_data

    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)

    controlled_joint_indices = []
    for idx in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, idx)
        name = info[1].decode()
        if name in {f"joint{i}" for i in range(1, 7)}:
            controlled_joint_indices.append(idx)
        elif name == "drive_joint":
            p.resetJointState(robot_id, idx, 0.45)
    controlled_joint_indices.sort()
    for idx, q in zip(controlled_joint_indices, q0):
        p.resetJointState(robot_id, idx, float(q))

    obstacle_positions, obstacle_radii = obstacles
    for pos, radius in zip(obstacle_positions, obstacle_radii):
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[0.1, 0.35, 1.0, 0.55])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(v) for v in pos],
        )
    return p, robot_id, controlled_joint_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--sim-urdf",
        type=Path,
        default=None,
        help="PyBullet visualization URDF. Defaults to arm-only, or gripper URDF with --with-gripper.",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Show the xArm gripper in PyBullet. OSCBF still controls only the 6 arm joints.",
    )
    parser.add_argument("--regen-urdf", action="store_true")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--no-gui", dest="gui", action="store_false")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--start-q", default="0,-0.7,-1.0,0,0.8,0")
    parser.add_argument("--goal", default="0.55,0.15,0.45")
    parser.add_argument("--num-obs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--kp-task", type=float, default=1.4)
    parser.add_argument("--kp-posture", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--qdot-limit", type=float, default=0.7)
    parser.add_argument("--safety-margin", type=float, default=0.015)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=0.02)
    args = parser.parse_args()

    require_dependencies()

    import jax
    import jax.numpy as jnp
    from cbfpy import CBF
    from oscbf.core.manipulator import Manipulator
    from oscbf.core.controllers import PoseTaskVelocityController

    q0 = parse_vec(args.start_q, 6, "start-q")
    goal = parse_vec(args.goal, 3, "goal")
    control_urdf_path = generate_uf850_urdf(args.urdf, force=args.regen_urdf, add_gripper=False)
    if args.sim_urdf is not None:
        sim_urdf_path = generate_uf850_urdf(args.sim_urdf, force=args.regen_urdf, add_gripper=args.with_gripper)
    elif args.with_gripper:
        sim_urdf_path = generate_uf850_urdf(DEFAULT_GRIPPER_URDF, force=args.regen_urdf, add_gripper=True)
    else:
        sim_urdf_path = control_urdf_path
    obstacle_positions, obstacle_radii = sample_obstacles(args.num_obs, args.seed)

    collision_model = make_uf850_collision_model(args.radius_scale, args.safety_margin)
    robot = Manipulator.from_urdf(str(control_urdf_path), collision_data=collision_model)
    if robot.num_joints != 6:
        raise RuntimeError(f"Expected 6 controlled joints from UF850 URDF, got {robot.num_joints}")

    config_cls = make_velocity_config_class()
    cbf_config = config_cls(robot, obstacle_positions, obstacle_radii, args.z_min, args.alpha)
    cbf = CBF.from_config(cbf_config)
    controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([args.kp_task] * 3 + [0.0, 0.0, 0.0]),
        kp_joint=args.kp_posture,
        qdot_min=-args.qdot_limit * np.ones(robot.num_joints),
        qdot_max=args.qdot_limit * np.ones(robot.num_joints),
    )

    p, robot_id, joint_indices = setup_pybullet(sim_urdf_path, q0, (obstacle_positions, obstacle_radii), args.gui)

    des_rot = np.eye(3)
    des_q = q0.copy()
    zero3 = np.zeros(3)

    @jax.jit
    def compute_control(q):
        m_inv, jac, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]
        u_nom = controller(
            q,
            pos,
            rot,
            jnp.asarray(goal),
            jnp.asarray(des_rot),
            jnp.asarray(zero3),
            jnp.asarray(zero3),
            jnp.asarray(des_q),
            jac,
            m_inv,
        )
        return cbf.safety_filter(q, u_nom)

    q = jnp.asarray(q0)
    min_clearance = float("inf")
    for step in range(args.steps):
        qdot = np.asarray(compute_control(q), dtype=float)
        q = jnp.asarray(np.asarray(q) + args.dt * qdot)
        for idx, q_i in zip(joint_indices, np.asarray(q)):
            p.resetJointState(robot_id, idx, float(q_i))
        p.stepSimulation()
        if args.gui:
            time.sleep(args.dt)

        ee_pos = np.asarray(robot.ee_position(q))
        collision = np.asarray(robot.link_collision_data(q))
        if obstacle_positions.size:
            deltas = collision[:, None, :3] - obstacle_positions[None, :, :]
            clearances = np.linalg.norm(deltas, axis=2) - collision[:, None, 3] - obstacle_radii[None, :]
            min_clearance = min(min_clearance, float(np.min(clearances)))

        if step % 120 == 0:
            dist = float(np.linalg.norm(ee_pos - goal))
            print(f"step={step:04d} ee={ee_pos.round(3)} dist={dist:.3f} min_clearance={min_clearance:.3f}")
        if np.linalg.norm(ee_pos - goal) < 0.025:
            print(f"Reached goal at step {step}")
            break

    print(f"Final q={np.asarray(q).round(4)} min_clearance={min_clearance:.4f}")
    if args.gui:
        input("Press Enter to close PyBullet...")
    p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
