"""Reusable, ROS-free UF850 kinematic simulation and control loop."""

from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np
import pybullet as p

from xarm_density.nominal import PoseReference
from xarm_density.robot import RobotWorldConfig, UF850CDFController
from xarm_density.config import load_config, validate_config

ASSETS = Path(__file__).resolve().parent / "assets"


def load_scene(path=None):
    return load_config(path)


class Simulation:
    """Joint-velocity integration, not a motor-dynamics or hardware interface."""

    def __init__(self, scene=None, hz=None, gui=False, backend=None):
        if hz is not None and (not np.isfinite(hz) or hz <= 0):
            raise ValueError("Control rate must be finite and positive")
        self.scene = load_scene() if scene is None else validate_config(scene)
        if hz is not None:
            self.scene['dt_s'] = 1 / hz
        if backend is not None:
            self.scene['qp_backend'] = backend
        self.scene = validate_config(self.scene)
        scene = self.scene
        self.dt, self.time, self.reference_start = scene['dt_s'], 0., 0.
        config = RobotWorldConfig(
            q_start=tuple(scene["q_start"]), goal=tuple(scene["goal"]),
            obstacle_centers=tuple(map(tuple, scene["obstacle_centers"])),
            obstacle_radii=tuple(scene["obstacle_radii"]), safety_mode="whole_body",
            sensing_width=scene["sensing_width"], residual_margin=scene["residual_margin"],
            qdot_limit=scene["qdot_limit"], include_gripper=True,
            gripper_opening=scene["gripper_opening"], tcp_offset=tuple(scene['tcp_offset']), dt=self.dt,
            alpha=scene['alpha'], v_epsilon=scene['v_epsilon'],
            target_radius=scene['target_radius'], ee_radius=scene['ee_radius'],
            nominal_speed=scene['nominal_speed'], nominal_gain=scene['nominal_gain'],
            task_weight=scene['task_weight'], cartesian_task_weight=scene['cartesian_task_weight'],
            input_weight=scene['input_weight'], theta_weight=scene['theta_weight'],
            density_floor=scene['density_floor'], basis_scale=tuple(scene['basis_scale']),
            collision_padding=scene['collision_padding'], solver_settings=scene['solver_settings'],
        )
        self.controller = UF850CDFController(config, qp_backend=scene['qp_backend'])
        self.q = np.asarray(scene["q_start"], dtype=float)
        robot = self.controller.robot
        if np.any(self.q < np.asarray(robot.joint_lower_limits)) or np.any(self.q > np.asarray(robot.joint_upper_limits)):
            raise ValueError('q_start exceeds the URDF joint limits')
        self.reference = PoseReference(self.controller.robot, scene, self.q)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Cannot connect to PyBullet; use --headless without a desktop")
        self.robot_id = p.loadURDF(str(ASSETS / "uf850/gripper.urdf"), useFixedBase=True, physicsClientId=self.client)
        self.arm_joints, self.gripper_joints, self.joint_names = [], [], []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            if info[1].decode() in {f"joint{k}" for k in range(1,7)}:
                self.arm_joints.append(j)
            elif info[2] != p.JOINT_FIXED:
                self.gripper_joints.append(j)
            if info[2] != p.JOINT_FIXED:
                self.joint_names.append(info[1].decode())
        self.arm_joints.sort()
        self.obstacles = []
        for center, radius in zip(scene["obstacle_centers"], scene["obstacle_radii"]):
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=self.client)
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=(.6,.63,.66,1), physicsClientId=self.client)
            self.obstacles.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visual, basePosition=center,
                                                    physicsClientId=self.client))
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=.018, rgbaColor=(0,.6,.3,1), physicsClientId=self.client)
        self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                         basePosition=scene["goal"], physicsClientId=self.client)
        p.resetDebugVisualizerCamera(scene['camera_distance'], scene['camera_yaw'],
                                    scene['camera_pitch'], scene['camera_target'], physicsClientId=self.client)
        self._display()
        if self.controller.whole_body_clearance(self.q) <= 0:
            self.close()
            raise ValueError("Starting configuration collides with the modeled obstacles")
        # Compile and initialize both control and diagnostic kernels before timing.
        for _ in range(scene['warmup_steps']):
            self.controller.solve(self.q, self.reference.command(self.q, 0.))
        np.asarray(self.controller.ee_position(jnp.asarray(self.q)))
        self.records = []

    def _display(self):
        for joint, value in zip(self.arm_joints, self.q):
            p.resetJointState(self.robot_id, joint, float(value), physicsClientId=self.client)
        for joint in self.gripper_joints:
            p.resetJointState(self.robot_id, joint, self.scene["gripper_opening"], physicsClientId=self.client)
        p.performCollisionDetection(physicsClientId=self.client)

    def set_goal(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)) or np.linalg.norm(goal) > self.scene['goal_envelope_radius']:
            raise ValueError("Goal must be three finite coordinates within the configured preview envelope")
        for center, radius in zip(self.scene["obstacle_centers"], self.scene["obstacle_radii"]):
            if np.linalg.norm(goal-center) <= radius + self.scene['ee_radius']:
                raise ValueError("Goal is inside the preliminary tool exclusion region")
        self.reference.start = np.asarray(self.controller.ee_position(jnp.asarray(self.q)))
        self.reference.goal = goal
        self.controller.target = jnp.asarray(goal)
        self.reference_start = self.time
        p.resetBasePositionAndOrientation(self.goal_id, goal, [0,0,0,1], physicsClientId=self.client)

    def step(self):
        start = perf_counter()
        nominal = self.reference.command(self.q, self.time-self.reference_start)
        result = self.controller.solve(self.q, nominal)
        command_elapsed = perf_counter()-start
        u = np.asarray(result["input"])
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > self.scene["qdot_limit"] + self.scene['command_tolerance']:
            raise RuntimeError("Invalid joint command")
        if result["density_floor_active"] or result["residual"] < self.scene["residual_margin"]-self.scene['residual_tolerance']:
            raise RuntimeError("Density floor or infeasible transport residual; stopping simulation")
        next_q = self.q + self.dt*u
        robot = self.controller.robot
        if np.any(next_q < np.asarray(robot.joint_lower_limits)) or np.any(next_q > np.asarray(robot.joint_upper_limits)):
            raise RuntimeError("Predicted joint position exceeds the URDF limits")
        self.q = next_q
        self.time += self.dt
        self._display()
        clearance, contacts = float("inf"), 0
        for obstacle in self.obstacles:
            values = p.getClosestPoints(self.robot_id, obstacle, 2., physicsClientId=self.client)
            if values:
                clearance = min(clearance, min(value[8] for value in values))
            contacts += len(p.getContactPoints(self.robot_id, obstacle, physicsClientId=self.client))
        position = np.asarray(self.controller.ee_position(jnp.asarray(self.q)))
        target, _ = self.reference.reference(self.time-self.reference_start)
        record = {"time_s": self.time, "q": self.q.copy(), "command": u.copy(),
                  "nominal": nominal, "tcp": position, "reference": target,
                  "target_error_m": float(np.linalg.norm(position-self.reference.goal)),
                  "tracking_error_m": float(np.linalg.norm(position-target)),
                  "sphere_clearance_m": self.controller.whole_body_clearance(self.q),
                  "mesh_clearance_m": clearance, "contacts": contacts,
                  "residual": result["residual"], "control_time_s": command_elapsed,
                  "solver_time_s": result["solver_solve_time"],
                  "loop_time_s": perf_counter()-start}
        self.records.append(record)
        if contacts or record["sphere_clearance_m"] <= 0:
            raise RuntimeError("Obstacle collision detected; stopping simulation")
        return record

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def summary(self):
        if not self.records:
            return {"steps": 0}
        times = np.array([r["control_time_s"] for r in self.records]) * 1000
        return {"steps": len(self.records), "simulated_time_s": self.time,
                "hz": 1/self.dt, "dt_ms": self.dt*1000,
                "final_target_error_m": self.records[-1]["target_error_m"],
                "target_reached": self.records[-1]["target_error_m"] <= self.scene['target_radius'],
                "minimum_sphere_clearance_m": min(r["sphere_clearance_m"] for r in self.records) if self.obstacles else None,
                "minimum_mesh_clearance_m": min(r["mesh_clearance_m"] for r in self.records) if self.obstacles else None,
                "contact_steps": sum(r["contacts"]>0 for r in self.records),
                "control_median_ms": float(np.median(times)),
                "control_p95_ms": float(np.percentile(times,95)),
                "control_max_ms": float(times.max()),
                "control_deadline_misses": int(np.sum(times>self.dt*1000))}
