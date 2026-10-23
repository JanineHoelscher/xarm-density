"""Run the density controller against a MoveIt fake arm via Servo JointJog."""

import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from control_msgs.msg import JointJog
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.srv import ServoCommandType
from rclpy.node import Node
from sensor_msgs.msg import JointState

from xarm_density.nominal import PoseReference
from xarm_density.robot import RobotWorldConfig, UF850CDFController


class CDFVelocityController(Node):
    def __init__(self):
        super().__init__('xarm_density_fake_controller')
        share = Path(get_package_share_directory('xarm_density_ros'))
        self.declare_parameter('scene', str(share / 'scene.json'))
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_jog_topic', '/servo_server/delta_joint_cmds')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('max_joint_speed', 0.35)
        scene = json.loads(Path(self.get_parameter('scene').value).read_text())
        self.names = [f'joint{i}' for i in range(1, 7)]
        speed = min(float(self.get_parameter('max_joint_speed').value),
                    float(scene['qdot_limit']), 0.7)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('max_joint_speed must be finite and positive')
        self.config = RobotWorldConfig(
            q_start=tuple(scene['q_start']), goal=tuple(scene['goal']),
            obstacle_centers=tuple(map(tuple, scene['obstacle_centers'])),
            obstacle_radii=tuple(scene['obstacle_radii']), safety_mode='whole_body',
            sensing_width=scene['sensing_width'], residual_margin=scene['residual_margin'],
            qdot_limit=speed, include_gripper=True,
            gripper_opening=scene['gripper_opening'], tcp_offset=(0.0, 0.0, 0.172),
        )
        self.scene = scene
        self.speed = speed
        self.controller = None
        self.reference = None
        self.start_time = None
        self.jog = self.create_publisher(
            JointJog, self.get_parameter('joint_jog_topic').value, 10)
        self.servo_type = self.create_client(
            ServoCommandType, '/servo_server/switch_command_type')
        self._servo_type_future = None
        self._servo_type_ready = False
        self._warned_missing_service = False
        self.subscription = self.create_subscription(
            JointState, self.get_parameter('joint_state_topic').value,
            self.on_joint_state, 10)
        hz = float(self.get_parameter('rate_hz').value)
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('rate_hz must be finite and positive')
        self.timer = self.create_timer(1.0 / hz, self.update)
        self.q = None
        self.last_stamp = None
        self.failed = False
        self.get_logger().info(
            f'Fake-arm density control ready; outputting at {hz:g} Hz, '
            f'joint speed limited to {speed:g} rad/s')

    def on_joint_state(self, message):
        indices = {name: i for i, name in enumerate(message.name)}
        if any(name not in indices for name in self.names):
            return
        q = np.asarray([message.position[indices[name]] for name in self.names], dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        if self.last_stamp == stamp:
            return
        self.last_stamp = stamp
        self.q = q

    def publish_velocity(self, values):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'link_base'
        msg.joint_names = self.names
        msg.velocities = [float(value) for value in values]
        self.jog.publish(msg)

    def stop(self, reason):
        self.publish_velocity(np.zeros(6))
        if not self.failed:
            self.get_logger().error(f'Stopping fake-arm controller: {reason}')
        self.failed = True
        self.timer.cancel()

    def update(self):
        if self.q is None or self.failed:
            return
        if not self._ensure_joint_jog_mode():
            return
        try:
            if self.controller is None:
                self.controller = UF850CDFController(self.config)
                self.reference = PoseReference(self.controller.robot, self.scene, self.q)
                self.start_time = time.monotonic()
                self.get_logger().info('Joint feedback received; compiling controller')
            elapsed = time.monotonic() - self.start_time
            nominal = self.reference.command(self.q, elapsed)
            result = self.controller.solve(self.q, nominal)
            velocity = np.asarray(result['input'], dtype=float)
            if (not np.all(np.isfinite(velocity))
                    or np.max(np.abs(velocity)) > self.speed + 1e-6
                    or result['density_floor_active']
                    or result['residual'] < self.scene['residual_margin'] - 1e-6):
                self.stop('invalid command, density floor, or infeasible CDF residual')
                return
            self.publish_velocity(velocity)
        except Exception as exc:
            self.stop(str(exc))

    def _ensure_joint_jog_mode(self):
        if self._servo_type_ready:
            return True
        if self._servo_type_future is not None:
            if not self._servo_type_future.done():
                return False
            try:
                response = self._servo_type_future.result()
            except Exception as exc:
                self.get_logger().error(f'Could not select Servo joint-jog mode: {exc}')
                self._servo_type_future = None
                return False
            self._servo_type_future = None
            if not response.success:
                self.get_logger().error('MoveIt Servo rejected joint-jog mode')
                return False
            self._servo_type_ready = True
            self.get_logger().info('MoveIt Servo is accepting joint-jog commands')
            return True
        if not self.servo_type.service_is_ready():
            if not self._warned_missing_service:
                self.get_logger().info('Waiting for MoveIt Servo command-type service')
                self._warned_missing_service = True
            return False
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.JOINT_JOG
        self._servo_type_future = self.servo_type.call_async(request)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = CDFVelocityController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok() and not node.failed:
            node.publish_velocity(np.zeros(6))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
