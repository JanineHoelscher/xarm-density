import json
import socket
import time

import rclpy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState

from xarm_density_ros.node import SimulationNode, valid_state


def packet():
    return dict(version=1, sent_at=time.monotonic(), names=[f'joint{k}' for k in range(1, 7)],
                positions=[0.] * 6, tcp=[.3, .2, .4])


def test_validation():
    state = packet()
    assert valid_state(state)
    state['positions'][0] = float('nan')
    assert not valid_state(state)
    state = packet()
    state['sent_at'] -= 2
    assert not valid_state(state)
    assert not valid_state({})


def test_ros_state_and_goal_roundtrip():
    rclpy.init()
    bridge = SimulationNode()
    observer = rclpy.create_node('density_bridge_test')
    received = []
    subscription = observer.create_subscription(JointState, '/joint_states', received.append, 10)
    publisher = observer.create_publisher(PointStamped, '/xarm_density/goal', 10)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(('127.0.0.1', 15051))
    udp.setblocking(False)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not received:
            udp.sendto(json.dumps(packet()).encode(), ('127.0.0.1', 15050))
            rclpy.spin_once(bridge, timeout_sec=.02)
            rclpy.spin_once(observer, timeout_sec=.02)
        assert received, 'No ROS JointState from UDP state'
        assert received[-1].name == packet()['names']
        goal = PointStamped()
        goal.header.frame_id = 'world'
        goal.point.x, goal.point.y, goal.point.z = .38, .35, .43
        command = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and command is None:
            udp.sendto(json.dumps(packet()).encode(), ('127.0.0.1', 15050))
            publisher.publish(goal)
            rclpy.spin_once(bridge, timeout_sec=.02)
            rclpy.spin_once(observer, timeout_sec=.02)
            try:
                command = json.loads(udp.recvfrom(4096)[0])
            except BlockingIOError:
                pass
        assert command is not None, 'No UDP goal from ROS PointStamped'
        assert command['position'] == [.38, .35, .43]
        assert command['frame'] == 'world'
    finally:
        udp.close()
        observer.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()
