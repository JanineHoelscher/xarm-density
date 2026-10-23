"""Publish simulator feedback, forward validated point goals, never robot commands."""

import json
import math
import signal
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def valid_state(message):
    try:
        names, positions = message['names'], message['positions']
        age = time.monotonic() - message['sent_at']
        return (message['version'] == 1 and 0 <= age < 1
                and isinstance(names, list) and 6 <= len(names) <= 32
                and all(isinstance(n, str) for n in names) and len(set(names)) == len(names)
                and all(f'joint{k}' in names for k in range(1, 7))
                and isinstance(positions, list) and len(positions) == len(names)
                and isinstance(message['tcp'], list) and len(message['tcp']) == 3
                and all(type(v) in (int, float) and math.isfinite(v)
                        for v in positions + message['tcp']))
    except (KeyError, TypeError, ValueError):
        return False


class SimulationNode(Node):
    def __init__(self):
        super().__init__('xarm_density_simulation_bridge')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 15050))
        self.socket.setblocking(False)
        self.joints = self.create_publisher(JointState, '/joint_states', 10)
        self.tcp = self.create_publisher(PointStamped, '/xarm_density/tcp', 10)
        self.status = self.create_publisher(String, '/xarm_density/status', 10)
        self.goal_subscriber = self.create_subscription(PointStamped, '/xarm_density/goal', self.goal, 10)
        self.last_state = None
        self.timer = self.create_timer(.02, self.poll)
        self.get_logger().info('SIMULATION ONLY: waiting for xarm-sim --ros; no hardware command publisher')

    def goal(self, message):
        values = [message.point.x, message.point.y, message.point.z]
        if message.header.frame_id != 'world' or not all(math.isfinite(v) for v in values):
            self.get_logger().warning('Rejected goal: use finite coordinates in world frame')
            return
        if self.last_state is None or time.monotonic() - self.last_state > 1:
            self.get_logger().warning('Rejected goal: simulator feedback is stale or missing')
            return
        packet = dict(version=1, frame='world', sent_at=time.monotonic(), position=values)
        self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), ('127.0.0.1', 15051))

    def poll(self):
        latest = None
        for _ in range(64):
            try:
                data, _ = self.socket.recvfrom(16384)
            except BlockingIOError:
                break
            try:
                message = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            if valid_state(message):
                latest = message
        if latest is not None:
            self.last_state = time.monotonic()
            stamp = self.get_clock().now().to_msg()
            joints = JointState()
            joints.header.stamp = stamp
            joints.name, joints.position = latest['names'], latest['positions']
            self.joints.publish(joints)
            tcp = PointStamped()
            tcp.header.stamp, tcp.header.frame_id = stamp, 'world'
            tcp.point.x, tcp.point.y, tcp.point.z = map(float, latest['tcp'])
            self.tcp.publish(tcp)
        state = 'running' if self.last_state is not None and time.monotonic()-self.last_state < 1 else 'waiting_or_stale'
        self.status.publish(String(data=state))

    def destroy_node(self):
        self.socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
