# Validation 10: Live UF850 CDF-QP and GPU optimization

Status: planned; implementation and benchmarks have not started.

## Objective

Make validation 9 interactive and determine whether GPU acceleration improves
performance. Preserve its UF850, closed gripper, TCP trajectory, five obstacles,
limits, density settings, and numerical precision for the default comparison.

## Baseline and profiling

- [ ] Profile validation 9 and preserve its existing results.
- [ ] Separate startup/JIT compilation from steady-state performance.
- [ ] Measure kinematics, density/derivatives, QP assembly, solver, CPU/GPU
      transfers, collision queries, and rendering individually.
- [ ] Report median, 95th percentile, maximum latency, and missed deadlines.

## GPU environment

- [ ] Check RTX 2070 and driver compatibility with supported JAX/CUDA builds.
- [ ] Install CUDA-enabled JAX in `xarm` and pin reproducible dependencies.
- [ ] Verify actual computation placement, not just GPU discovery.
- [ ] Support explicit CPU/GPU selection and report the selected device.

Clarabel runs on CPU. GPU JAX initially accelerates kinematics, density, and
derivatives only. Transfers and dispatch can outweigh gains for small problems;
select the default backend using measurements.

## Numerical optimization

- [ ] JIT numerical kernels and warm up before starting motion.
- [ ] Batch link-sphere/obstacle evaluations.
- [ ] Keep array shapes stable; use obstacle capacity and masks if needed.
- [ ] Avoid recompilation when goals or obstacle positions change.
- [ ] Reuse QP structure and solver state where supported.
- [ ] Remove redundant calculations and minimize CPU/GPU transfers.
- [ ] Synchronize device work explicitly when measuring GPU execution time.
- [ ] Check precision, trajectories, feasibility, and clearance after changes.
- [ ] Consider a GPU-native QP solver only if profiling justifies it.

## Live PyBullet example

- [ ] Add `scripts/pybullet_validation_10.py`, sharing existing controller code.
- [ ] Open a GUI with validation 9 as the default scene.
- [ ] Add pause/resume and reset.
- [ ] Add an editable Cartesian goal and obstacle selection/position controls.
- [ ] Toggle collision spheres and the reference trajectory.
- [ ] Handle edits causing collision or infeasibility explicitly.
- [ ] Display target error, clearance, solver status, latency, and simulation speed.
- [ ] Provide headless mode and optional recording.

## Control and visualization timing

- [ ] Separate rendering at about 30--60 FPS from controller updates.
- [ ] Choose a sustainable control rate from measured controller latency.
- [ ] Define timestep, wall-clock pacing, and missed-deadline behavior.
- [ ] Keep recording and diagnostics from dominating control time.

Retain kinematic joint-velocity integration and joint-state playback for this
phase. Motor-driven dynamics and tracking-error validation are separate work.

## Benchmarks and verification

- [ ] Compare CPU/GPU headlessly with identical settings and precision.
- [ ] Repeat with GUI enabled and recording disabled.
- [ ] Compare convergence, trajectory/orientation error, sphere and Bullet mesh
      clearance, contacts, CDF residual, QP feasibility, and input limits.
- [ ] Save timing distributions, configuration, device/dependency information,
      plots, and metrics; report compilation separately.
- [ ] Report whether execution meets the chosen wall-clock deadline.
- [ ] Choose the default backend from measured results.

## Intended interface

These commands are planned; the validation 10 script is not implemented yet.

```bash
conda activate xarm
cd /home/sriramk/xarm_ws
python scripts/pybullet_validation_10.py --device gpu --gui
python scripts/pybullet_validation_10.py --device cpu --headless --benchmark
python scripts/pybullet_validation_10.py --device gpu --headless --benchmark
```

## Completion criteria

- [ ] Responsive live demo with documented controls and installation steps.
- [ ] Default scene matches validation 9 within documented numerical tolerances.
- [ ] Default scene reaches the target without obstacle contacts and respects limits.
- [ ] Reproducible CPU/GPU comparison; any speedup is demonstrated.
- [ ] Real-time claims reflect measured latency, not simulated timestep alone.
- [ ] Add a scenario README and link results from the parent overview.
