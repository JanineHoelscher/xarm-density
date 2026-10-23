# xarm-density

Whole-arm density-based motion control for the UFactory UF850. One Python
entry point loads the arm with its gripper, five spherical obstacles, a
Cartesian pose nominal controller, and the optimized seven-coordinate CDF-QP.
The default control rate is 100 Hz (10 ms).

Start with the [code and workflow guide](docs/CODE_EXPLAINER.md) for a file map,
the simulation/control flow, and the current ROS interface versus the future
hardware integration steps.

Paper sources, figures, and saved CSVs are tracked under
[scripts/latex](scripts/latex/README.md) and [scripts/figures](scripts/figures/README.md)
for rebuilding the paper and replotting its recorded numerical results.

## Standalone setup: no ROS required

```bash
git clone https://github.com/sriram-2502/xarm-density.git
cd xarm-density
conda env create -f environment.yml
conda activate xarm
python run_simulation.py
```

Run from a clean shell without sourcing a ROS installation. This script opens
PyBullet and runs the seven-second clutter scene. Initial JAX compilation may
take a few seconds. The GUI requires a desktop/OpenGL display.

`run_simulation.py` shows the experiment workflow explicitly: display settings,
open the simulator, run the control loop, save results, and close resources.
Shared helpers live in `src/xarm_density/cli.py`; the nominal-control/CDF-QP/
joint-integration step is in `src/xarm_density/simulation.py`. The installed
`xarm-sim` command uses the same helpers and accepts the same arguments. Run the
script from the repository root after activating the environment above.

```bash
python run_simulation.py --help
python run_simulation.py --headless --quiet
```

The launcher prints the chosen config, rate, start, target, progress, and a
short result summary. `--quiet` suppresses periodic progress only. Interrupted
or failed runs save partial results with `run_status: stopped` once the
simulator is initialized; successful runs use `run_status: completed`.

Headless, configurable rate, and a short dependency smoke test:

```bash
python run_simulation.py --headless --hz 100
python run_simulation.py --headless --hz 50
python run_simulation.py --headless --steps 20 --output results/smoke
```

The GUI is paced to wall time; headless runs as fast as possible. Outputs are
`results/latest/metrics.json`, `scene.json`, and raw `trajectory.csv`. Change
the output path to keep separate runs. CPU is the tested default backend.

The default scene was verified in a fresh isolated environment, both headless
and with the GUI. See [standalone verification](docs/STANDALONE_VERIFICATION.md)
for tested rates, clearances, timing, limitations, and local CSV evidence.

For an existing environment or ordinary Python virtual environment:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Python 3.10--3.12 is supported by the declared package requirements; the clean
reference environment uses Python 3.11. Numerical dependencies are pinned in
`pyproject.toml`. Binary Python packages are installed during setup; the repo
is not an offline wheelhouse and is not literally dependency-free.

## Change the scene

```bash
python run_simulation.py --headless --goal 0.38 0.35 0.43
python run_simulation.py --config path/to/my_experiment.json
```

Use `src/xarm_density/assets/scene.json` as the configuration example. It
contains simulation timestep/duration, QP weights and solver tolerances,
start joints, target position, obstacle centers/radii, reference duration,
gains, density settings, and camera settings. See
[configuration reference](docs/CONFIGURATION.md) for all keys and units.
Set `dt_s` to `0.01` for 100 Hz or `0.02` for 50 Hz. Positions are meters;
angles are radians. Arbitrary goals are not guaranteed reachable or navigable
by this local controller.

Run without arguments to use this config automatically. Custom configs can
override a subset of keys; command-line options override config values. Each
successful run saves the effective `config.json` alongside its CSV and metrics.

## What is included

- Optimized CDF-QP, full reference solver, smooth density window, and nominal.
- Attributed OSCBF numerical kinematics/control subset, included locally.
- UF850 arm/gripper URDFs and all referenced meshes, included locally.
- Checked kinematics descriptor retaining the original fixed-link merging.
- Tests for QP equivalence, packaged assets, and headless simulation.

No external `density-drift`, `oscbf`, `xarm_ros2`, xacro generation, or absolute
home-directory paths are needed to launch the standalone simulation. See
[THIRD_PARTY.md](THIRD_PARTY.md) for origins and licenses and
[docs/CDF_QP.md](docs/CDF_QP.md) for the mathematical formulation.

## ROS 2 Jazzy and hardware direction

The optional ROS package publishes simulated joints/TCP and accepts point
goals, without MoveIt or Gazebo. Keep conda and ROS in **separate terminals**;
do not mix apt-installed `rclpy` into conda. Jazzy's binary platform is Ubuntu
24.04. Follow [docs/ROS2_JAZZY.md](docs/ROS2_JAZZY.md) for the complete setup.

```bash
# Native ROS terminal, after installing Jazzy and colcon:
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2 --packages-select xarm_density_ros
source install/setup.bash
ros2 launch xarm_density_ros simulation.launch.py

# Separate clean conda terminal:
conda activate xarm
python run_simulation.py --ros --steps 6000 --output results/ros-demo
```

The standalone tests and Jazzy container build/integration tests
[pass in CI](https://github.com/sriram-2502/xarm-density/actions/runs/40002248041).
This development host runs Ubuntu 22.04/Humble; native Jazzy is not locally
installed. The bridge is simulation-only and never sends physical robot commands.

The current backend performs joint-velocity integration and joint-state
playback, not motor dynamics. It stops on failed solves, density-floor use,
invalid commands, predicted joint-position limit crossings, and detected
obstacle collisions. These checks do not establish hardware safety. Measured
state feedback, command watchdogs, acceleration limits, robot-specific velocity
interfaces, and hardware validation are still needed before physical execution.

## Local archive

Superseded ROS/Gazebo loaders, experiments, workspace builds, and historical
copies are preserved under ignored `archive/`. Active paper sources, figure
exports, plotting helpers, and recorded CSV inputs are tracked under `scripts/`.
The remaining archived files are
not needed by this package and are not distributed in a fresh clone. Previously
published versions remain in Git history. The research paper remains available
as [density-xarm.pdf](density-xarm.pdf).
