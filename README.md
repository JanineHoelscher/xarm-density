# xarm-density

Whole-arm density-based motion control for the UFactory UF850. One Python
entry point loads the arm with its gripper, five spherical obstacles, a
Cartesian pose nominal controller, and the optimized seven-coordinate CDF-QP.
The default control rate is 100 Hz (10 ms).

## Standalone setup: no ROS required

```bash
git clone https://github.com/sriram-2502/xarm-density.git
cd xarm-density
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
xarm-sim
```

Run from a clean shell without sourcing a ROS installation. `xarm-sim` opens
PyBullet and runs the seven-second clutter scene. Initial JAX compilation may
take a few seconds. The GUI requires a desktop/OpenGL display.

Headless, configurable rate, and a short dependency smoke test:

```bash
xarm-sim --headless --hz 100
xarm-sim --headless --hz 50
xarm-sim --headless --steps 20 --output results/smoke
```

The GUI is paced to wall time; headless runs as fast as possible. Outputs are
`results/latest/metrics.json`, `scene.json`, and raw `trajectory.csv`. Change
the output path to keep separate runs. CPU is the tested default backend.

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
xarm-sim --headless --goal 0.38 0.35 0.43
xarm-sim --scene path/to/scene.json
```

Use `src/xarm_density/assets/scene.json` as the configuration example. It
contains start joints, target position, obstacle centers/radii, reference
duration, gains, sensing width, and joint-rate bounds. Positions are meters;
angles are radians. Arbitrary goals are not guaranteed reachable or navigable
by this local controller.

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
goals, without MoveIt or Gazebo. Keep the Python virtual environment and ROS
setup in **separate terminals**; the ROS bridge needs the apt-installed `rclpy`
from the native ROS environment. Jazzy's binary platform is Ubuntu
24.04. Follow [docs/ROS2_JAZZY.md](docs/ROS2_JAZZY.md) for the complete setup.

```bash
# Native ROS terminal, after installing Jazzy and colcon:
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2 --packages-select xarm_density_ros
source install/setup.bash
ros2 launch xarm_density_ros simulation.launch.py

# Python terminal with the project virtual environment activated:
xarm-sim --ros --steps 6000 --output results/ros-demo
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

Superseded ROS/Gazebo loaders, experiments, paper source, plots, recorded runs,
and vendor workspace builds are preserved under ignored `archive/`. They are
not needed by this package and are not distributed in a fresh clone. Previously
published versions remain in Git history. The research paper remains available
as [density-xarm.pdf](density-xarm.pdf).
# Fake UF850 arm in MoveIt / RViz

The ROS integration can run the CDF controller against UFACTORY's fake
`ros2_control` arm. It sends joint velocity jogs into MoveIt Servo, which then
drives the fake trajectory controller. This starts no physical robot driver.

From the ROS workspace root (`~/Code/ros2_ws`), build this package after the
`xarm_ros2` packages are available in the workspace:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to xarm_density_ros
```

In a new terminal, start it from the `xarm-density` directory:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Code/ros2_ws/install/setup.bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch xarm_density_ros fake_moveit.launch.py
```

RViz opens with the fake UF850 and the density controller starts once it sees
`/joint_states`. The default controller scene is installed from
`src/xarm_density/assets/scene.json`. To stop, press Ctrl+C in the launch
terminal. MoveIt Servo's input timeout also halts the fake controller if
velocity messages stop arriving.

This path requires the ROS 2 Jazzy `xarm_ros2` MoveIt packages, MoveIt Servo,
RViz, and the ROS messages used by the launch. The Python environment needs the
project requirements installed. Activate `.venv` before launching so the
controller process uses that Python while inheriting ROS's sourced environment.
