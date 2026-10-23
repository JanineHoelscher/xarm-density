import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from xarm_density.model import parse_urdf
from xarm_density.robot import DEFAULT_URDF
from xarm_density.simulation import ASSETS, Simulation


class StandaloneTests(unittest.TestCase):
    def test_relative_meshes_exist(self):
        for name in ("arm.urdf", "gripper.urdf"):
            path = ASSETS / "uf850" / name
            root = ET.parse(path).getroot()
            self.assertIsNone(root.find("ros2_control"))
            for mesh in root.findall(".//mesh"):
                filename = mesh.attrib["filename"]
                self.assertFalse(Path(filename).is_absolute())
                self.assertTrue((path.parent / filename).is_file())

    def test_descriptor(self):
        self.assertEqual(parse_urdf(DEFAULT_URDF)["num_joints"], 6)

    def test_headless_and_runtime_goal(self):
        sim = Simulation(gui=False)
        try:
            for _ in range(20):
                record = sim.step()
                self.assertEqual(record["contacts"], 0)
                self.assertGreater(record["sphere_clearance_m"], 0)
                self.assertGreaterEqual(record["residual"], .01 - 1e-6)
            sim.set_goal([.39,.35,.43])
            np.testing.assert_allclose(sim.controller.target, [.39,.35,.43])
            self.assertEqual(sim.step()["contacts"], 0)
            with self.assertRaises(ValueError):
                sim.set_goal([float("nan"), 0, 0])
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
