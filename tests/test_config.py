import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from xarm_density.config import load_config, validate_config
from xarm_density.simulation import Simulation


class ConfigTests(unittest.TestCase):
    def test_defaults_and_legacy_partial_config(self):
        config = validate_config({'kp_position': 2.0})
        self.assertEqual(config['kp_position'], 2.0)
        self.assertEqual(config['dt_s'], .01)
        self.assertEqual(config['input_weight'], .02)
        self.assertEqual(config['theta_weight'], .0002)
        self.assertEqual(load_config()['qp_backend'], 'reduced')

    def test_invalid_config(self):
        for changes in [{'dt_s': 0}, {'alpha': float('nan')}, {'q_start': [0]*5},
                        {'goal': [0, 0]}, {'theta_weight': 0}, {'input_weight': -1},
                        {'warmup_steps': 0}, {'qdot_limit': .8}, {'task_weigth': 1},
                        {'basis_scale': [0]*6}, {'solver_settings': {'max_iter': 0}},
                        {'solver_settings': {'typo': 1}}, {'duration_s': .001},
                        {'obstacle_centers': [[0, 0, 0]], 'obstacle_radii': []},
                        {'solver_settings': {'tol_feas': float('inf')}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(changes)

    def test_no_obstacles_valid(self):
        config = validate_config({'obstacle_centers': [], 'obstacle_radii': []})
        self.assertEqual(config['obstacle_centers'], [])
        simulation = Simulation(config)
        try:
            simulation.step()
            self.assertIsNone(simulation.summary()['minimum_sphere_clearance_m'])
            json.dumps(simulation.summary(), allow_nan=False)
        finally:
            simulation.close()

    def test_parameters_reach_controller(self):
        config = validate_config({'dt_s': .02, 'task_weight': 2., 'input_weight': .03,
                                  'theta_weight': .0003, 'alpha': .25, 'kp_joint': .3,
                                  'target_radius': 1., 'solver_settings': {'max_iter': 150}})
        simulation = Simulation(config)
        try:
            self.assertEqual(simulation.dt, .02)
            self.assertEqual(simulation.controller.config.task_weight, 2.)
            self.assertEqual(simulation.controller.config.input_weight, .03)
            self.assertEqual(simulation.controller.config.theta_weight, .0003)
            self.assertEqual(simulation.controller.config.alpha, .25)
            self.assertEqual(simulation.controller.qp.controller.config.solver_settings['max_iter'], 150)
            record = simulation.step()
            np.testing.assert_allclose(record['q'], np.asarray(config['q_start']) + .02*record['command'])
            self.assertTrue(simulation.summary()['target_reached'])
        finally:
            simulation.close()

    def test_cli_config_and_overrides_are_saved(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / 'custom.json'
            config.write_text(json.dumps({'dt_s': .02, 'duration_s': .04, 'task_weight': 1.5}))
            command = [sys.executable, str(root/'run_simulation.py'), '--headless',
                       '--config', str(config), '--output', str(path/'configured')]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            metrics = json.loads((path/'configured/metrics.json').read_text())
            self.assertEqual(metrics['steps'], 2)
            self.assertEqual(metrics['hz'], 50.)
            effective = json.loads((path/'configured/config.json').read_text())
            self.assertEqual(effective['task_weight'], 1.5)
            subprocess.run(command + ['--hz', '100', '--steps', '3', '--output', str(path/'override')],
                           check=True, capture_output=True, text=True, timeout=60)
            metrics = json.loads((path/'override/metrics.json').read_text())
            effective = json.loads((path/'override/config.json').read_text())
            self.assertEqual(metrics['steps'], 3)
            self.assertEqual(effective['dt_s'], .01)
            self.assertEqual(effective['duration_s'], .03)
