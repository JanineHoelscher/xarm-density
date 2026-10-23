# Simulation and controller configuration

The default editable config is [`src/xarm_density/assets/scene.json`](../src/xarm_density/assets/scene.json).
It is included in the installed package. With the editable installation from
`environment.yml`, changing this file changes the next run without reinstalling.
No additional YAML/TOML dependency is needed.

```bash
conda activate xarm
python run_simulation.py
python run_simulation.py --config src/xarm_density/assets/scene.json
python run_simulation.py --config path/to/my_experiment.json --headless
```

A custom JSON config can contain only the keys you change; missing keys come
from the bundled default. Unknown keys, invalid shapes, nonfinite values,
negative weights, and invalid solver settings are rejected before simulation.
`--scene` remains an alias for `--config`, so older scene files still load.

## Time and solver

| Key | Meaning / units | Default |
| --- | --- | --- |
| `dt_s` | Joint integration/control timestep, seconds; rate = 1/dt | 0.01 (100 Hz) |
| `duration_s` | Total simulated run length, seconds | 7.0 |
| `reference_duration_s` | Minimum-jerk reference travel time, seconds | 7.0 |
| `qp_backend` | Exact reduced QP or full reference QP | `reduced` |
| `warmup_steps` | Untimed initialization solves before the loop | 3 |
| `solver_settings.max_iter` | Clarabel iteration limit | 200 |
| `solver_settings.tol_gap_abs` | Absolute gap tolerance | 1e-8 |
| `solver_settings.tol_gap_rel` | Relative gap tolerance | 1e-8 |
| `solver_settings.tol_feas` | Feasibility tolerance | 1e-8 |

For 50 Hz set `dt_s` to `0.02`. For a longer hold after reaching the reference
endpoint, increase `duration_s` while retaining `reference_duration_s`.
The run uses `round(duration_s / dt_s)` steps. This is a **kinematic integration
timestep**, not a PyBullet motor-physics timestep. GUI runs are paced; ordinary
headless runs are not. Solver tolerances are not physical clearance margins.

## Start, target, and geometry

| Key | Meaning / units |
| --- | --- |
| `q_start` | Six initial joint angles, radians, ordered joint1 through joint6; checked against URDF limits |
| `goal` | Desired final TCP `[x,y,z]`, meters, in the robot `world` frame |
| `obstacle_centers` | List of spherical obstacle `[x,y,z]` centers, meters |
| `obstacle_radii` | Matching positive obstacle radii, meters; use two empty lists for no obstacles |
| `tcp_offset` | TCP translation from the modeled flange frame, meters; default `[0,0,0.172]` |
| `gripper_opening` | Fixed gripper joint angle, radians, range [0,0.85]; 0.85 is the tested closed cover |
| `collision_padding` | Nonnegative extra radius on every robot collision sphere, meters; default 0 |
| `target_radius` | Target-reached reporting tolerance, meters; default 0.035 |
| `goal_envelope_radius` | Runtime ROS target preview envelope about world origin, meters; default 1 |
| `ee_radius` | Preliminary runtime target/obstacle exclusion radius, meters; default 0.055 |

The starting TCP is computed by forward kinematics from `q_start`; it is not
an independent Cartesian start. End-effector orientation is held at its initial
value. A different Cartesian start requires an appropriate six-joint start
configuration; this file does not run IK. Changing TCP offset does not change
the visual gripper mesh. Arbitrary tool geometry must be modeled/validated
separately. Link-sphere centers, radii, URDF limits, and robot dimensions remain
model code/assets, not arbitrary experiment settings. This controller handles
sphere obstacles; cuboids/cylinders are not supported by this config yet.

## Nominal control and CDF-QP

| Key | Meaning | Default |
| --- | --- | --- |
| `kp_position` | Operational-space position feedback gain | 1.4 |
| `kp_orientation` | Operational-space orientation feedback gain | 1.0 |
| `kp_joint` | Nominal joint posture gain; no redundant nullspace for this 6-DOF pose task | 0.2 |
| `qdot_limit` | Symmetric joint-rate bound, rad/s; must be <= 0.7 | 0.7 |
| `task_weight` | Weight on tracking nominal joint velocities in the playground | 1.0 |
| `cartesian_task_weight` | Tracking weight for the separate direct-Cartesian fallback API | 8.0 |
| `input_weight` | Joint-velocity squared regularization | 0.02 |
| `theta_weight` | Positive flux-coefficient squared regularization | 0.0002 |
| `sensing_width` | Smooth obstacle density transition width, meters | 0.1 |
| `alpha` | Target-distance density exponent | 0.2 |
| `v_epsilon` | Squared-distance denominator regularizer, m^2 | 0.001 |
| `density_floor` | Numerical density division floor; activation stops the loop | 1e-10 |
| `residual_margin` | Required pointwise density-transport residual | 0.01 |
| `basis_scale` | Six joint-angle scales, radians, for quadratic monomials | pi per joint |
| `nominal_gain`, `nominal_speed` | Direct-Cartesian fallback gain and speed cap (m/s); not used by the pose nominal | 1.4, 0.22 |

With the usual nominal joint velocity, the QP objective is:

```math
\min_\theta\;\frac{w_t}{2}\|B(q)\theta-u_{nom}\|^2
+\frac{w_u}{2}\|B(q)\theta\|^2+\frac{w_\theta}{2}\|\theta\|^2,
\quad a(q)^T\theta\ge\gamma,\quad
-u_{max}\le B(q)\theta\le u_{max}.
```

Here `task_weight` is w_t, `input_weight` is w_u, `theta_weight` is w_theta,
`residual_margin` is gamma, and `qdot_limit` is u_max. The basis degree stays
fixed at two; this is required by the implemented controller, not a degree
selection feature. See [CDF_QP.md](CDF_QP.md) for the density/flux derivation.

`command_tolerance` and `residual_tolerance` (both default 1e-6) are numerical
post-solve acceptance tolerances; they are not obstacle safety margins. Avoid
relaxing them to hide failed solves. Larger `collision_padding` is the explicit
way to inflate robot geometry, but feasibility and convergence can change.

## View, overrides, and saved settings

`camera_distance` is meters; `camera_yaw` and `camera_pitch` are degrees;
`camera_target` is a world-frame point in meters. These only affect the GUI.

Command-line `--hz`, `--steps`, `--goal`, and `--backend` take precedence over
config values. GUI/headless mode, pacing, ROS transport, and output path remain
command-line options. For example:

```bash
python run_simulation.py --config path/to/my_experiment.json --hz 50 --steps 350
```

Each successful run saves the resolved parameters **after overrides** to
`config.json` (and the backwards-compatible `scene.json`) beside `metrics.json`
and raw `trajectory.csv`. Runtime ROS retargets are not an event log in this
initial config snapshot. Configuration changes are not proof of feasibility,
reachability, gripper coverage, self-collision avoidance, or hardware safety.
