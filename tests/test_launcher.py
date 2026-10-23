from contextlib import contextmanager, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from xarm_density import cli


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('readable_launcher', ROOT / 'run_simulation.py')
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class LauncherTests(unittest.TestCase):
    def test_nonfinite_rate_rejected(self):
        for value in ('nan', 'inf', '-1', '0'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.parse_arguments(['--hz', value])
                self.assertEqual(error.exception.code, 2)

    def test_interrupted_workflows_save_partial_results(self):
        for module in (cli, LAUNCHER):
            with self.subTest(workflow=module.__name__), tempfile.TemporaryDirectory() as directory:
                close = Mock()
                simulation = SimpleNamespace(records=[], summary=lambda: {'steps': 0})

                @contextmanager
                def session(config, options):
                    try:
                        yield simulation, None
                    finally:
                        close()

                config = {'dt_s': .01, 'duration_s': 7.}
                with patch.object(module, 'simulation_session', session), \
                        patch.object(module, 'print_run_settings'), \
                        patch.object(module, 'run_control_loop', side_effect=KeyboardInterrupt), \
                        patch.object(cli, 'load_run_config', return_value=config), \
                        redirect_stderr(io.StringIO()):
                    result = cli.run_cli(['--headless', '--output', directory], workflow=module.run_experiment)
                self.assertEqual(result, 130)
                close.assert_called_once()
                metrics = json.loads((Path(directory) / 'metrics.json').read_text())
                self.assertEqual(metrics['run_status'], 'stopped')
                self.assertEqual(metrics['requested_steps'], 700)
                self.assertTrue((Path(directory) / 'trajectory.csv').exists())

    def test_bridge_start_failure_closes_simulation(self):
        options = SimpleNamespace(headless=True, ros=True)
        simulation = Mock()
        with patch('xarm_density.simulation.Simulation', return_value=simulation), \
                patch('xarm_density.bridge.SimulationBridge', side_effect=OSError('port in use')):
            with self.assertRaises(OSError):
                with cli.simulation_session({}, options):
                    self.fail('Session should not open when bridge initialization fails')
        simulation.close.assert_called_once()
