#!/usr/bin/env python3
"""Gazebo-only phase-1 UF850 safety-planner harness.

This intentionally skips MoveIt. It starts the UF850 Gazebo + ros2_control
stack, then runs a Python node that:

- waits for /joint_states;
- waits for the UF850 trajectory controller topic;
- instantiates a basic planner shell;
- optionally publishes a tiny joint-space demo trajectory.

The next planner phase will replace the demo trajectory with a CDF/CBF-style QP
that computes qdot_safe, then sends q_next = q + dt * qdot_safe to the same
controller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Iterable


XARM_WS = Path("/home/sriramk/xarm_ws")
SETUP_BASH = XARM_WS / "install" / "setup.bash"

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
TRAJ_TOPIC = "/uf850_traj_controller/joint_trajectory"


def _shell_with_setup(command: str) -> list[str]:
    if not SETUP_BASH.exists():
        raise FileNotFoundError(f"ROS setup file not found: {SETUP_BASH}")
    return ["bash", "-lc", f"source {shlex.quote(str(SETUP_BASH))} && {command}"]


def _terminate(process: subprocess.Popen, name: str, timeout: float = 8.0) -> None:
    if process.poll() is not None:
        return
    print(f"[phase1] stopping {name} (pid={process.pid})")
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()


def run_parent(args: argparse.Namespace) -> int:
    processes: list[tuple[str, subprocess.Popen]] = []
    script_path = Path(__file__).resolve()

    try:
        if args.launch != "none":
            launch_cmd = (
                "ros2 launch xarm_gazebo _robot_beside_table_gazebo.launch.py "
                "dof:=6 robot_type:=uf850 hw_ns:=ufactory "
                "load_controller:=true show_rviz:=false no_gui_ctrl:=true "
                f"gz_type:={shlex.quote(args.gz_type)}"
            )
            print(f"[phase1] starting Gazebo/ros2_control: {launch_cmd}")
            launch_proc = subprocess.Popen(_shell_with_setup(launch_cmd))
            processes.append(("uf850_gazebo", launch_proc))
            time.sleep(args.launch_settle_sec)

        child_args = [
            shlex.quote(sys.executable),
            shlex.quote(str(script_path)),
            "--child-node",
            "--status-period-sec",
            str(args.status_period_sec),
            "--ready-timeout-sec",
            str(args.ready_timeout_sec),
        ]
        if args.once:
            child_args.append("--once")
        if args.demo_motion:
            child_args.append("--demo-motion")
            child_args.extend(["--demo-joint", args.demo_joint])
            child_args.extend(["--demo-amplitude", str(args.demo_amplitude)])
            child_args.extend(["--demo-duration-sec", str(args.demo_duration_sec)])
        child_cmd = " ".join(child_args)
        print(f"[phase1] starting planner harness node: {child_cmd}")
        child_proc = subprocess.Popen(_shell_with_setup(child_cmd))
        processes.append(("planner_harness", child_proc))

        if args.once or args.demo_motion:
            return child_proc.wait()

        print("[phase1] running. Press Ctrl-C to stop both processes.")
        while True:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[phase1] {name} exited with code {code}")
                    return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[phase1] interrupted")
        return 130
    finally:
        for name, proc in reversed(processes):
            _terminate(proc, name)


@dataclass
class SafetyPlannerConfig:
    expected_joints: tuple[str, ...] = JOINT_NAMES
    traj_topic: str = TRAJ_TOPIC
    control_dt: float = 0.02
    max_joint_speed: float = 0.25


class PhaseOneSafetyPlanner:
    """Planner shell for Gazebo-only testing.

    This is where the CDF/CBF QP will live next. Phase 1 only stores the robot
    state and provides a tiny bounded demo command.
    """

    def __init__(self, config: SafetyPlannerConfig):
        self.config = config
        self.latest_positions: dict[str, float] = {}

    def update_joint_state(self, names: Iterable[str], positions: Iterable[float]) -> None:
        self.latest_positions.update(zip(names, positions))

    @property
    def missing_joints(self) -> list[str]:
        return [name for name in self.config.expected_joints if name not in self.latest_positions]

    @property
    def ready(self) -> bool:
        return not self.missing_joints

    def current_q(self) -> list[float]:
        return [self.latest_positions[name] for name in self.config.expected_joints]

    def tiny_demo_target(self, joint_name: str, amplitude: float, phase: float) -> list[float]:
        q = self.current_q()
        if joint_name not in self.config.expected_joints:
            raise ValueError(f"Unknown demo joint {joint_name}; expected one of {self.config.expected_joints}")
        idx = self.config.expected_joints.index(joint_name)
        q[idx] += float(amplitude) * math.sin(phase)
        return q

    def describe(self, controller_seen: bool) -> str:
        available = [name for name in self.config.expected_joints if name in self.latest_positions]
        return (
            f"joints_seen={available} missing={self.missing_joints} "
            f"traj_topic={self.config.traj_topic} controller_seen={controller_seen}"
        )


def run_child_node(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    except ImportError as exc:
        print("[phase1] ROS 2 Python imports failed.")
        print(f"[phase1] {exc}")
        print(f"[phase1] Try: source {SETUP_BASH}")
        return 2

    class PlannerHarnessNode(Node):
        def __init__(self) -> None:
            super().__init__("uf850_gazebo_safety_planner_harness")
            self.planner = PhaseOneSafetyPlanner(SafetyPlannerConfig())
            self.publisher = self.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)
            self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
            self.create_timer(args.status_period_sec, self._status_tick)
            self._reported_ready = False
            self.get_logger().info("UF850 Gazebo safety-planner harness started")

        def _on_joint_state(self, msg: JointState) -> None:
            self.planner.update_joint_state(msg.name, msg.position)

        def _controller_seen(self) -> bool:
            for topic_name, topic_types in self.get_topic_names_and_types():
                if topic_name == TRAJ_TOPIC and "trajectory_msgs/msg/JointTrajectory" in topic_types:
                    return True
            return False

        def ready(self) -> bool:
            return self.planner.ready and self._controller_seen()

        def _status_tick(self) -> None:
            controller_seen = self._controller_seen()
            self.get_logger().info(self.planner.describe(controller_seen))
            if self.ready() and not self._reported_ready:
                self._reported_ready = True
                self.get_logger().info(
                    "Phase 1 ready: joint states and uf850_traj_controller command topic are visible."
                )

        def publish_position_target(self, positions: list[float], time_from_start_sec: float) -> None:
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = list(JOINT_NAMES)
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.time_from_start = Duration(seconds=float(time_from_start_sec)).to_msg()
            msg.points = [point]
            self.publisher.publish(msg)

    rclpy.init(args=None)
    node = PlannerHarnessNode()
    try:
        deadline = time.monotonic() + args.ready_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.ready():
            node.get_logger().error("Timed out waiting for /joint_states and trajectory controller topic")
            return 1
        node.get_logger().info("Readiness check passed")

        if args.demo_motion:
            node.get_logger().info(
                f"Running tiny demo on {args.demo_joint}: amplitude={args.demo_amplitude:.3f} rad"
            )
            start = time.monotonic()
            rate_sec = 0.25
            while rclpy.ok():
                elapsed = time.monotonic() - start
                if elapsed > args.demo_duration_sec:
                    break
                phase = 2.0 * math.pi * elapsed / max(args.demo_duration_sec, 1e-6)
                target = node.planner.tiny_demo_target(args.demo_joint, args.demo_amplitude, phase)
                node.publish_position_target(target, time_from_start_sec=rate_sec)
                end_wait = time.monotonic() + rate_sec
                while rclpy.ok() and time.monotonic() < end_wait:
                    rclpy.spin_once(node, timeout_sec=0.05)
            node.publish_position_target(node.planner.current_q(), time_from_start_sec=0.5)
            node.get_logger().info("Tiny demo complete")
            return 0

        if args.once:
            return 0

        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        choices=["gazebo", "none"],
        default="gazebo",
        help="Launch Gazebo/ros2_control before starting the planner harness.",
    )
    parser.add_argument(
        "--gz-type",
        choices=["gazebo", "gz", "ignition"],
        default="gazebo",
        help="Simulator backend passed to xarm_gazebo lower-level launch.",
    )
    parser.add_argument("--launch-settle-sec", type=float, default=10.0)
    parser.add_argument("--status-period-sec", type=float, default=2.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=45.0)
    parser.add_argument("--once", action="store_true", help="Exit after readiness check.")
    parser.add_argument("--demo-motion", action="store_true", help="Send a tiny bounded joint command after ready.")
    parser.add_argument("--demo-joint", choices=JOINT_NAMES, default="joint6")
    parser.add_argument("--demo-amplitude", type=float, default=0.04)
    parser.add_argument("--demo-duration-sec", type=float, default=6.0)
    parser.add_argument("--child-node", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child_node:
        return run_child_node(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

