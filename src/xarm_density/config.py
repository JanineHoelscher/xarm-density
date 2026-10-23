"""Load and validate the shared JSON simulation/controller configuration."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'assets/scene.json'


def validate_config(values):
    defaults = json.loads(DEFAULT_CONFIG.read_text())
    if not isinstance(values, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = deepcopy(defaults)
    config.update(deepcopy(values))
    solver = defaults['solver_settings'].copy()
    if not isinstance(config['solver_settings'], dict) or set(config['solver_settings']) - set(solver):
        raise ValueError('Unknown or invalid solver_settings')
    solver.update(config['solver_settings'])
    config['solver_settings'] = solver

    def vector(key, shape):
        array = np.asarray(config[key], dtype=float)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f'{key} must have shape {shape} and finite entries')
        config[key] = array.tolist()
        return array

    vector('q_start', (6,))
    vector('goal', (3,))
    vector('tcp_offset', (3,))
    vector('camera_target', (3,))
    if np.any(vector('basis_scale', (6,)) <= 0):
        raise ValueError('basis_scale entries must be positive')
    centers = np.asarray(config['obstacle_centers'], dtype=float)
    radii = np.asarray(config['obstacle_radii'], dtype=float)
    if centers.size == 0:
        centers = centers.reshape(0, 3)
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
        raise ValueError('Need matching obstacle_centers (N,3) and obstacle_radii (N,)')
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError('Obstacle geometry must be finite, with positive radii')
    config['obstacle_centers'], config['obstacle_radii'] = centers.tolist(), radii.tolist()
    nonnegative = {'residual_margin', 'input_weight', 'kp_position', 'kp_orientation',
                   'kp_joint', 'collision_padding'}
    positive = {'dt_s', 'duration_s', 'reference_duration_s', 'qdot_limit',
                'sensing_width', 'task_weight', 'cartesian_task_weight', 'theta_weight',
                'alpha', 'v_epsilon', 'density_floor', 'nominal_speed', 'nominal_gain',
                'target_radius', 'goal_envelope_radius', 'ee_radius', 'camera_distance',
                'command_tolerance', 'residual_tolerance'}
    for key in positive | nonnegative | {'camera_yaw', 'camera_pitch', 'gripper_opening'}:
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        if key in positive and value <= 0 or key in nonnegative and value < 0:
            raise ValueError(f'{key} is outside its allowed range')
    if config['qdot_limit'] > .7 or not 0 <= config['gripper_opening'] <= .85:
        raise ValueError('qdot_limit must be <= 0.7 rad/s; gripper_opening must be in [0, 0.85]')
    if config['qp_backend'] not in ('reduced', 'full'):
        raise ValueError('qp_backend must be reduced or full')
    if type(config['warmup_steps']) is not int or config['warmup_steps'] < 1:
        raise ValueError('warmup_steps must be a positive integer')
    if type(solver['max_iter']) is not int or solver['max_iter'] < 1:
        raise ValueError('solver max_iter must be a positive integer')
    for key in ('tol_gap_abs', 'tol_gap_rel', 'tol_feas'):
        if type(solver[key]) not in (int, float) or not np.isfinite(solver[key]) or solver[key] <= 0:
            raise ValueError(f'solver {key} must be finite and positive')
    if config['duration_s'] < config['dt_s']:
        raise ValueError('duration_s must be at least dt_s')
    return config


def load_config(path=None):
    return validate_config(json.loads(Path(path or DEFAULT_CONFIG).read_text()))
