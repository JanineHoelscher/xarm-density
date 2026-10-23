import json
import socket
import time
import unittest

from xarm_density.bridge import SimulationBridge


class BridgeTests(unittest.TestCase):
    def test_goal_validation_and_freshness(self):
        bridge = SimulationBridge(state_port=15050, goal_port=0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = bridge.socket.getsockname()
        def send(packet):
            sender.sendto(json.dumps(packet).encode(), address)
        try:
            send(dict(version=1, frame='world', sent_at=time.monotonic(), position=[.3, .2, .4]))
            self.assertEqual(bridge.receive_goal(), [.3, .2, .4])
            send(dict(version=1, frame='world', sent_at=time.monotonic()-2, position=[.3, .2, .4]))
            self.assertIsNone(bridge.receive_goal())
            send(dict(version=1, frame='camera', sent_at=time.monotonic(), position=[.3, .2, .4]))
            self.assertIsNone(bridge.receive_goal())
            send(dict(version=1, frame='world', sent_at=time.monotonic(), position=[float('nan'), .2, .4]))
            self.assertIsNone(bridge.receive_goal())
        finally:
            sender.close()
            bridge.close()
