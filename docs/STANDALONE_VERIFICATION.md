# Standalone simulation verification

Verified on 2026-10-23 using a copy of the current source tree in `/tmp`,
excluding `.git`, the local archive, previous results, and workspace builds.
The copy included the new `run_simulation.py` launcher. A fresh Python 3.11
virtual environment installed only the package and its declared dependencies
with `python -m pip install -e .`. Inherited Python/ROS import paths were cleared.
The existing conda environment and external research packages were not used as
installed dependencies. This verifies the current working files, including
the launcher, rather than claiming those uncommitted files are on GitHub.

## Checks

- All 10 unit tests passed: QP equivalence, packaged mesh references, model
  descriptor, short simulation, runtime target update, and optional bridge input
  validation. The bridge test uses localhost sockets but does not require ROS.
- `pip check` reported no broken requirements.
- `oscbf`, `cbfpy`, `density_drift`, and `rclpy` were absent from the environment.
- The simulation Python source contained no absolute home-directory references
  or runtime imports of those external packages.
- The default Python launcher completed full headless runs at 100 and 50 Hz.
- The default GUI completed the full 700-step run with a desktop OpenGL 3.3
  context on the RTX 2070 SUPER. The controller itself ran on CPU.
- The installed `xarm-sim` shortcut ran from outside the source directory,
  with a custom scene path, changed goal, and the full reference QP backend
  (20-step smoke test, not a full changed-goal tracking validation).
  That reference backend missed the 10 ms control budget on 3 of 20 steps
  (maximum 12.809 ms). Use the default reduced backend for the tested 100 Hz path.

## Default seven-second scene

| Mode | Steps | Target error | Min sphere clearance | Min mesh clearance | Contact steps | Median control | P95 control | Max control | Control deadline misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Headless, 100 Hz | 700 | 4.758 mm | 20.414 mm | 48.487 mm | 0 | 0.948 ms | 1.046 ms | 1.407 ms | 0 |
| Headless, 50 Hz | 350 | 4.621 mm | 20.491 mm | 48.396 mm | 0 | 0.927 ms | 1.057 ms | 1.367 ms | 0 |
| GUI, 100 Hz | 700 | 4.758 mm | 20.414 mm | 48.487 mm | 0 | 0.979 ms | 1.125 ms | 1.531 ms | 0 |

All three runs reached the configured 35 mm target tolerance. These are single
runs, not a repeatability study or worst-case real-time guarantee. Control
timing covers nominal control plus the CDF solve after warmup; it excludes
rendering, collision diagnostics, and pacing. CSV also records loop timing.

## Evidence and reproduction

Raw `trajectory.csv`, `scene.json`, and `metrics.json` are preserved locally
under `results/standalone_verification/2026-10-23/`, in `100hz`, `50hz`,
`gui100hz`, and `full-backend-smoke` subfolders. `packages.json` records the
fresh environment and `cpu.json` the processor details. Results are gitignored;
a fresh clone reproduces them by running the commands below, not by downloading
this local evidence.

After the setup in the main README:

```bash
python -m unittest discover -s tests -v
python -m pip check
python run_simulation.py --headless --hz 100 --output results/verify/100hz
python run_simulation.py --headless --hz 50 --output results/verify/50hz
python run_simulation.py --hz 100 --output results/verify/gui100hz
```

## Scope and known warnings

The UF850/gripper models and required numerical subset from OSCBF are bundled;
no separate vendor/research checkout is required. Initial dependency installation
still downloads Python packages, and the GUI needs a working desktop/OpenGL.

PyBullet warns about missing inertial data on fixed `link_eef` and `link_tcp`
frames. This does not prevent the tested kinematic playback; it must not be
interpreted as validating a motor-dynamics simulation. This verification does
not cover arbitrary obstacle layouts, arbitrary reachable goals, self-collision
avoidance, ROS hardware execution, or physical safety. The backend integrates
joint velocities and resets simulated joint positions; it is not a motor model.
