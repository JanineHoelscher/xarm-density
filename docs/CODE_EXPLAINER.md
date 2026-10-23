# Code and workflow guide

This guide follows the code that runs today, then outlines how to extend it
toward ROS-controlled hardware. The important distinction is:

**The current controller computes joint velocities. PyBullet displays their
kinematic integration. ROS can exchange simulation feedback and goals, but
there is no physical robot command interface in this repository yet.**

## 1. Start here

From the repository root, with the environment already created:

```bash
conda activate xarm
python run_simulation.py
```

Useful alternatives:

```bash
python run_simulation.py --help
python run_simulation.py --headless --output results/my_run
python run_simulation.py --config my_experiment.json --hz 50
```

The installed `xarm-sim` command runs the same shared controller loop. The root
script is the readable experiment entry point; the installed command has its
own small orchestration function using the same helpers.

## 2. Which file does what?

| File | Responsibility |
| --- | --- |
| [run_simulation.py](../run_simulation.py) | Readable experiment workflow: show settings, open resources, run, save, close |
| [cli.py](../src/xarm_density/cli.py) | Arguments, numerical setup, config overrides, resource management, loop, progress, result files |
| [config.py](../src/xarm_density/config.py) | Merge defaults with a custom JSON file and validate keys, shapes, values, and solver settings |
| [scene.json](../src/xarm_density/assets/scene.json) | Default timestep, scene, gains, density parameters, QP weights, tolerances, and view |
| [simulation.py](../src/xarm_density/simulation.py) | Robot/obstacle loading and one complete nominal-control/CDF-QP/integration/diagnostic step |
| [nominal.py](../src/xarm_density/nominal.py) | Minimum-jerk TCP reference and operational-space nominal joint velocity |
| [robot.py](../src/xarm_density/robot.py) | Robot collision spheres, density, differentiated flux basis, QP inputs, clearance calculations |
| [core.py](../src/xarm_density/core.py) | Quadratic basis, command/transport matrices, full QP objective, exact reduced QP |
| [qp.py](../src/xarm_density/qp.py) | Cached Clarabel problem assembly, updates, solving, and solver status |
| [density.py](../src/xarm_density/density.py) | Smooth obstacle-window function |
| [model.py](../src/xarm_density/model.py) | Load the bundled numerical robot descriptor and check its matching URDF hash |
| [_vendor/oscbf](../src/xarm_density/_vendor/oscbf) | Included robot kinematics/dynamics and nominal-control code, not the OSCBF safety filter |
| [bridge.py](../src/xarm_density/bridge.py) | Optional simulator-side localhost transport; no ROS imports |
| [ROS node](../ros2/xarm_density_ros/xarm_density_ros/node.py) | Native ROS feedback publishers and goal subscriber |
| [ROS launch](../ros2/xarm_density_ros/launch/simulation.launch.py) | Start the ROS bridge and robot_state_publisher with the bundled gripper model |

Read the first seven rows in order to understand a normal simulation run.
Read `core.py` and `qp.py` next for the optimization details.

## 3. Startup workflow

```text
run_simulation.py: main()
  -> cli.run_cli(workflow=run_experiment)
     -> parse_arguments()
     -> configure_numerics()
     -> load_run_config()
     -> root script: run_experiment(config, options)
        -> print_run_settings()
        -> simulation_session()
           -> Simulation(config)
           -> optional SimulationBridge()
        -> run_control_loop()
        -> save_results()
        -> print_summary()
        -> close simulator and optional bridge
```

`configure_numerics()` runs before the heavy numerical imports: it limits
numerical-library threads and defaults JAX to CPU. This avoids accidental
thread oversubscription in this small control problem. PyBullet may still use
the GPU for GUI rendering; that is different from JAX controller computation.

Configuration precedence is: bundled defaults, custom JSON keys, then command
line overrides. For example, `--hz 50` overrides `dt_s` with 0.02 seconds.
The chosen duration is rounded to a whole number of steps.

`Simulation.__init__()` builds the controller, checks the initial joint limits,
loads the arm/gripper and sphere obstacles, displays the start, checks initial
modeled obstacle clearance, and warms up the nominal/CDF calculations. Initial
compilation is outside the reported per-step controller times.

## 4. What happens at each step?

At default settings, one control step represents 0.01 simulated seconds:

```text
optional new ROS goal
    |
current simulated q
    -> evaluate desired TCP position and velocity at reference time
    -> compute nominal joint velocity u_nom
    -> evaluate whole-arm density and flux derivatives at q
    -> solve CDF-QP for filtered joint velocity u
    -> check command, density floor, residual, and predicted joint limits
    -> integrate q_next = q + dt*u
    -> reset PyBullet joints to q_next
    -> compute tracking error, sphere clearance, mesh clearance, contacts
    -> append a trajectory record; stop if an obstacle collision is detected
    -> optionally publish feedback to ROS
```

The control calculations live in `Simulation.step()`. The outer
`run_control_loop()` handles iteration count, optional ROS goal updates,
progress, and wall-clock pacing. The GUI is paced unless `--fast` is used;
ordinary headless runs execute as fast as possible. ROS mode is always paced.

There is no `stepSimulation()` motor-dynamics loop here. Joint positions are
integrated in our code and assigned with PyBullet's `resetJointState()`, then
collision detection is performed. Good tracking therefore does not demonstrate
that a real actuator can follow the requested velocities or accelerations.

### Nominal controller: where does the desired motion come from?

`PoseReference` computes the starting TCP by forward kinematics from `q_start`.
The goal is the configured Cartesian TCP position; the desired orientation
stays at the initial orientation. A smooth reference connects the endpoints:

```math
s=\operatorname{clip}(t/T,0,1),\qquad
b(s)=10s^3-15s^4+6s^5,\qquad
x_d(t)=x_0+b(s)(x_g-x_0).
```

The reference derivative provides a desired TCP velocity. Position/orientation
feedback produces a desired six-dimensional task velocity. The bundled pose
controller maps it into joint velocities using a mass-weighted generalized
Jacobian inverse and clips to the configured nominal joint-rate limits.

**This is a velocity controller, not a torque-output inverse-dynamics
controller, and not a full-path IK planner.** The mass matrix weights the
velocity mapping; the output is still radians/second. For this six-joint,
six-dimensional pose task, there is no redundant posture nullspace.

### Whole-arm density: why are the links considered?

`make_uf850_collision_model()` defines multiple spheres on the arm and gripper.
Forward kinematics maps their centers into the world frame. For robot sphere
a and obstacle sphere b:

```math
d_{ab}(q)=\|p_a(q)-o_b\|-r_a-R_b.
```

Positive distance means separation for that sphere pair. A smooth window is
zero at collision and one outside the sensing width. The controller combines
all modeled pairs:

```math
\psi(q)=\prod_{a,b}S(d_{ab}(q)/\delta),\qquad
\rho(q)=\frac{\psi(q)}{(\|x_{TCP}(q)-x_g\|^2+\epsilon)^\alpha}.
```

Thus moving an elbow toward an obstacle changes the density even if the TCP
is far away. This is not TCP-only obstacle avoidance. The sphere geometry
must still cover the actual robot/tool appropriately; a positive modeled
clearance is only as meaningful as that geometry. PyBullet mesh clearance is
an additional diagnostic, not another constraint in the QP.

### CDF-QP: what exactly is optimized?

The implementation uses 28 monomials of total degree at most two in six
scaled joint coordinates. Multiplying by the obstacle window gives the scalar
flux basis w(q). Each of six velocity channels has its own 28 coefficients:
168 coefficients in total.

```math
v_i(q)=w(q)^T\theta_i,\qquad
u_i(q)=v_i(q)/\rho(q),\qquad u=B(q)\theta.
```

`pointwise_transport_row()` stacks the relevant derivatives of w to obtain
the divergence row a(q). The usual joint-velocity tracking problem is:

```math
\begin{aligned}
\min_\theta\quad &
\frac{w_t}{2}\|B\theta-u_{nom}\|^2+
\frac{w_u}{2}\|B\theta\|^2+
\frac{w_\theta}{2}\|\theta\|^2\\
\text{subject to}\quad & a(q)^T\theta\ge\gamma,\\
&-u_{max}\le B(q)\theta\le u_{max}.
\end{aligned}
```

The three weights are `task_weight`, `input_weight`, and `theta_weight`;
gamma is `residual_margin`. All sphere/obstacle pairs influence the density,
but the current formulation has **one combined pointwise transport inequality**,
not one inequality per pair. It is not a CBF constraint.

The reduced solver computes an orthonormal basis for the row space of B and
a. With positive isotropic coefficient regularization, unused orthogonal
coefficient components have optimal value zero. The same problem can therefore
be solved in seven coordinates (six command rows plus one transport row),
instead of 168 coefficients. The full backend remains available for comparison.
Clarabel caches the fixed-shape problem and updates its numerical matrices.

This code enforces an inequality at the sampled current state and recomputes
coefficients each step. It is not by itself a global density certificate or
a continuous-time/inter-sample collision guarantee. Read
[CDF_QP.md](CDF_QP.md) for the detailed mathematical formulation.

## 5. Outputs and failure behavior

The selected output folder contains:

| File | Contents |
| --- | --- |
| `config.json` | Resolved experiment settings after CLI overrides |
| `scene.json` | Backwards-compatible copy of the resolved config |
| `metrics.json` | Tracking/clearance/contact/timing summary, run status, requested steps |
| `trajectory.csv` | Recorded joints, nominal/filtered commands, TCP/reference positions, residuals, clearances, timings |

`run_status` is `completed` for a normally finished run, even if the target
was not reached; check `target_reached` separately. Expected controller/I/O
failures and keyboard interruption save partial results once the session is
initialized, with status `stopped`. The resource context closes the simulator
and optional bridge. Existing result files in the same directory are replaced.

Checks include command bounds/finiteness, density-floor activation, transport
residual, predicted joint-position limits, modeled obstacle clearance, and
mesh contact detection. A collision stop occurs after the kinematic step;
it is a failure detector, not a prediction that certifies the whole step safe.
Current logs are not a complete runtime ROS goal-event log.

Control timing covers nominal control plus the CDF solve after warmup.
Loop timing additionally includes integration and diagnostic work, but excludes
outer-loop ROS transport, progress printing, and sleeping. Neither metric alone
is an end-to-end hard real-time guarantee.

## 6. Using ROS today: no controller port needed

The optional ROS interface already exists. It deliberately uses two processes:

```text
Conda Python 3.11                         Native ROS Jazzy Python 3.12
-----------------------------------      --------------------------------
Simulation + JAX + Clarabel + PyBullet     xarm_density_ros SimulationNode
SimulationBridge                         robot_state_publisher

state JSON -> localhost UDP 15050 ------> JointState / TCP / status topics
goal JSON  <- localhost UDP 15051 <------ PointStamped goal subscriber
```

Keep these environments in separate terminals. Do not install apt `rclpy`
into conda. The ROS side does not import JAX or the numerical controller;
the conda side does not import `rclpy`.

On a Jazzy/Ubuntu 24.04 machine, build and launch from a native ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/xarm_ws
colcon build --base-paths ros2 --packages-select xarm_density_ros
source install/setup.bash
ros2 launch xarm_density_ros simulation.launch.py
```

In a separate clean conda terminal:

```bash
cd ~/xarm_ws
conda activate xarm
python run_simulation.py --ros --hz 100 --steps 6000 --output results/ros-demo
```

This runs for 60 simulated seconds. Startup may take a few seconds for JAX.
Use your actual clone path if it is not `~/xarm_ws`. See
[ROS2_JAZZY.md](ROS2_JAZZY.md) for prerequisite packages and RViz setup.

### ROS interfaces and a goal example

| Topic | Type | Direction at the ROS node |
| --- | --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` | Publishes simulated arm/gripper joint names and positions |
| `/xarm_density/tcp` | `geometry_msgs/msg/PointStamped` | Publishes simulated TCP position in `world` |
| `/xarm_density/status` | `std_msgs/msg/String` | Publishes `running` or `waiting_or_stale` |
| `/xarm_density/goal` | `geometry_msgs/msg/PointStamped` | Subscribes to desired TCP position in `world` |

From another terminal sourced with Jazzy and this workspace:

```bash
ros2 topic pub --once /xarm_density/goal geometry_msgs/msg/PointStamped \
  '{header: {frame_id: world}, point: {x: 0.38, y: 0.35, z: 0.43}}'
```

The node validates the frame, finite values, and presence of recent simulator
feedback. The simulator validates the goal preview envelope and preliminary
obstacle exclusion, starts a new reference at the current TCP, and updates the
density target without rebuilding the controller. The configured reference
duration is used again; orientation stays at its original desired value.

The ROS timer samples the latest state at up to 50 Hz, independently of the
100 Hz default controller. Timestamps are ROS wall-clock timestamps, not a
published simulation `/clock`; keep `use_sim_time=false`. UDP timestamps expire
after one second and require a shared host monotonic clock. Sequence numbers
are sent with state but are not used as an ordering/acknowledgment protocol.
Only one pair can bind the fixed ports. This is a localhost simulation
transport, not an authenticated robot-control network protocol.

If the ROS node stops, the simulator continues its current reference. If the
simulator stops, the ROS status becomes stale and new targets are rejected.
**That behavior is suitable for this optional simulation bridge, not a
physical robot watchdog.**

## 7. How to extend this toward hardware

The numerical controller is already separate from PyBullet. Keep it that way:
do not move simulation joint resets into a ROS callback or treat the simulated
`/joint_states` publisher as measured hardware feedback.

The proposed next architecture is **not implemented**:

```text
Native ROS/vendor driver measured joints
    -> validated, timestamped state transport
    -> conda controller: PoseReference + UF850CDFController.solve(q_measured, u_nom)
    -> filtered velocity plus state sequence/timestamp/status
    -> native ROS command gate/watchdog
    -> verified vendor velocity-control interface
    -> physical robot
```

1. Extract a control-step wrapper around the existing `PoseReference` and
   `UF850CDFController`. Inputs should be measured joint state, reference time,
   goal, and a validated obstacle snapshot; output should be joint velocity and
   diagnostics. Do not integrate a private q and use it as authoritative state.
2. Add a separate hardware ROS adapter, rather than repurposing
   `SimulationNode`. Subscribe to the vendor's measured state, verify joint
   ordering/units/frames and timestamps, and use the actual robot description
   and TCP calibration. Do not run competing simulated and measured publishers
   on the same state topic. Select the exact driver command API only after
   inspecting the installed Jazzy driver and enabled controller mode.
3. Replace the preview transport with a control-grade request/response
   contract: sequence-correlated measured state and command, expiry, startup
   handshake, enable/disable state, solver validity, and bounded processing.
   The current simulation packet schema is not enough for this.
4. Add hardware-specific joint velocity, acceleration, and predictive position
   bounds to the QP/command design. Do not independently clip or smooth the QP
   output afterward and assume the density inequality remains satisfied.
   Define and validate a robot-specific stop policy for stale state, missed
   deadlines, failed solves, missing perception, or transport loss.
5. Test a measured-state backend against a dynamic simulator/mock driver before
   physical execution. Recheck numerical behavior, communication timing,
   actuator tracking error, tool/sphere coverage, latency margins, workspace
   limits, and self-collision handling. Current obstacle density does not
   encode robot self-collision.
6. Stage physical tests with a separate emergency-stop path, explicit operator
   enable, conservative speed/acceleration limits, a verified stop procedure,
   and reviewed collision geometry. Passing the kinematic demo is not hardware
   readiness or a safety certification.

MoveIt is not required to evaluate this local controller. If introduced later,
keep its role explicit, such as providing a global reference or independent
scene checks; it does not automatically validate commands from this loop.

### Where camera obstacles would enter

Today obstacles are static spheres read at startup. The ROS bridge accepts
goals, not obstacle detections. Camera integration would require a perception
adapter that supplies geometry in the robot world frame, acquisition time,
and uncertainty, plus explicit stale-data behavior.

In the current controller, obstacle arrays are captured when the JAX kernels
are constructed. Editing `simulation.scene` during a run does not update those
kernels or the PyBullet obstacle bodies. Add and test a synchronized obstacle
update API (or controlled controller/scene reconstruction) before accepting
perception updates; a ROS subscription alone is not sufficient.

## 8. Tests and useful companion documents

```bash
conda activate xarm
python -m unittest discover -s tests -v
python run_simulation.py --headless --output results/regression
```

Tests cover full/reduced QP agreement, packaged assets, config validation and
override persistence, parameter propagation, short simulations, runtime goal
updates, launcher cleanup, and partial-result saving. ROS-side round-trip
tests are separate and run in the native ROS environment as described in
[ROS2_JAZZY.md](ROS2_JAZZY.md).

| Document | Use it for |
| --- | --- |
| [CONFIGURATION.md](CONFIGURATION.md) | Parameter meanings, units, defaults, and overrides |
| [CDF_QP.md](CDF_QP.md) | Density/flux and optimization mathematics |
| [ROS2_JAZZY.md](ROS2_JAZZY.md) | ROS installation prerequisites, build/run commands, topic checks |
| [STANDALONE_VERIFICATION.md](STANDALONE_VERIFICATION.md) | Recorded standalone verification and its scope |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Included upstream code/models and license attribution |
