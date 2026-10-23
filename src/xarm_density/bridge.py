"""Optional localhost-only simulation bridge; no ROS imports in this process."""

import json
import math
import socket
import time


class SimulationBridge:
    def __init__(self, state_port=15050, goal_port=15051):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', goal_port))
        self.socket.setblocking(False)
        self.destination = ('127.0.0.1', state_port)
        self.sequence = 0

    def receive_goal(self):
        latest = None
        for _ in range(32):
            try:
                data, _ = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                message = json.loads(data)
                goal = message['position']
                age = time.monotonic() - message['sent_at']
                if (message['version'] == 1 and message['frame'] == 'world'
                        and 0 <= age < 1 and isinstance(goal, list) and len(goal) == 3
                        and all(type(v) in (float, int) and math.isfinite(v) for v in goal)):
                    latest = goal
            except (ValueError, KeyError, TypeError):
                continue
        return latest

    def publish(self, simulation, record):
        positions = dict(zip((f'joint{k}' for k in range(1, 7)), record['q'].tolist()))
        positions.update({name: simulation.scene['gripper_opening']
                          for name in simulation.joint_names if name not in positions})
        packet = dict(version=1, sent_at=time.monotonic(), sequence=self.sequence,
                      names=list(positions), positions=list(positions.values()),
                      tcp=record['tcp'].tolist(), goal=simulation.reference.goal.tolist())
        self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), self.destination)
        self.sequence += 1

    def close(self):
        self.socket.close()
