# ROS 2 Jazzy: simulation bridge

This is an optional interface to the same standalone CDF-QP simulation.
It does not require MoveIt, Gazebo, xarm_ros2, a robot IP, or robot hardware.
No ROS node here sends joint commands to a physical arm.

## Environments

- **Controller:** conda `xarm-density`, Python 3.11, JAX/Clarabel/PyBullet.
- **ROS:** Ubuntu 24.04, ROS 2 Jazzy, system Python 3.12 and apt `rclpy`.

Use separate terminals/processes. Do not source ROS in the conda terminal or
install `rclpy` using pip. The two processes exchange small JSON datagrams on
localhost ports 15050 (state) and 15051 (goals). No third-party Python bridge
dependency is needed. Only one bridge/simulator pair can use these ports.

Jazzy's official binary platform is Ubuntu 24.04:
[installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
Our current development host is Ubuntu 22.04/Humble, so native Jazzy execution
has not been validated on that host. The supplied CI job builds/tests the ROS
package inside the official Jazzy/Noble container.

## 1. Build ROS side once

In a **fresh non-conda terminal** on Ubuntu 24.04 with Jazzy installed:

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-colcon-common-extensions ros-jazzy-robot-state-publisher
cd ~/xarm-density  # or your actual repository path, e.g. ~/xarm_ws
colcon build --base-paths ros2 --packages-select xarm_density_ros
source install/setup.bash
ros2 launch xarm_density_ros simulation.launch.py
```

Launch starts the UDP bridge and robot_state_publisher using the included
UF850-with-gripper model. It does not launch conda automatically.

## 2. Run simulation

In a separate **clean conda terminal**:

```bash
cd ~/xarm-density
conda activate xarm-density
xarm-sim --ros --steps 6000 --output results/ros-demo
```

This runs for 60 simulated seconds at 100 Hz, paced to wall time. Add
`--headless` to suppress the PyBullet window. Ordinary `xarm-sim` still works
without ROS and without opening any bridge socket. Compilation happens before
state streaming starts. `--fast` does not disable ROS-mode pacing.

The controller continues its current reference if ROS is absent; ROS is an
optional simulation interface, **not a hardware watchdog**. After the simulator
exits, the ROS status changes to `waiting_or_stale` within one second and goals
are rejected until feedback resumes. A stopped simulator must not be mistaken
for a running physical controller.

## 3. Inspect feedback and change target

In another sourced ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/xarm-density/install/setup.bash
ros2 topic echo /xarm_density/status
ros2 topic hz /joint_states
ros2 topic echo /xarm_density/tcp
ros2 topic pub --once /xarm_density/goal geometry_msgs/msg/PointStamped \
  '{header: {frame_id: world}, point: {x: 0.38, y: 0.35, z: 0.43}}'
```

Run each long-lived echo/hz command separately. Goals are meters in the URDF
`world` frame. A new target starts a smooth seven-second reference at the
current simulated TCP and updates the density target. Simulator-side checks
reject nonfinite targets, targets beyond the 1 m preview envelope, and targets
inside the preliminary tool/obstacle exclusion region. These checks are not
an IK reachability or path-existence guarantee. See the simulator terminal for
rejection messages.

`/joint_states` and TCP are published at up to 50 Hz (latest-state sampling),
independent of the 100 Hz controller. The node uses ROS wall-clock timestamps;
it does not publish `/clock`. Keep `use_sim_time=false`. For RViz, install
`ros-jazzy-rviz2`, use fixed frame `world`, and add RobotModel with description
topic `/robot_description`. The PyBullet window remains the obstacle view.

Packets use schema version 1, finite-value validation, bounded receive loops,
and a one-second monotonic timestamp expiry. Both processes must share the
same host monotonic clock (normal local terminals or host-network containers).
This is not an authenticated/network transport; do not expose these ports.

## Tests and portability

```bash
# Conda terminal
python -m unittest discover -s tests -v

# ROS terminal
colcon test --packages-select xarm_density_ros
colcon test-result --verbose
```

The GitHub Actions workflow separately checks standalone Python and Jazzy.
Until the workflow completes successfully, consider native Jazzy integration
unverified. No GPU is needed for the tested CPU path.

## Hardware work remaining

Keep the kinematic simulator as the regression test. Next implement a separate
measured-state backend using the vendor Jazzy driver, with joint names/frame
checks, velocity and acceleration limits, command watchdogs, an explicit enable
state, fault recovery, conservative geometry/calibration, and low-speed staged
tests. Do not replay these simulated joint states as hardware commands. CDF
residuals and simulation clearance alone are not a hardware safety guarantee;
self-collision is not currently part of the density product.
