"""Shared experiment helpers for run_simulation.py and the xarm-sim command."""

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, sleep


SCALAR_COLUMNS = (
    "time_s", "target_error_m", "tracking_error_m", "sphere_clearance_m",
    "mesh_clearance_m", "contacts", "residual", "control_time_s",
    "solver_time_s", "loop_time_s",
)
VECTOR_COLUMNS = {"q": 6, "command": 6, "nominal": 6, "tcp": 3, "reference": 3}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="UF850 whole-arm CDF-QP simulation (no hardware commands)",
        epilog="Defaults: src/xarm_density/assets/scene.json. CLI options override config values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "--scene", dest="config", type=Path,
                        help="JSON experiment config; otherwise use the bundled default")
    parser.add_argument("--headless", action="store_true", help="Run without the PyBullet window")
    parser.add_argument("--fast", action="store_true", help="Disable GUI pacing (ignored in ROS mode)")
    parser.add_argument("--hz", type=float, help="Override timestep with 1/hz")
    parser.add_argument("--steps", type=int, help="Override total run length in control steps")
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Override world-frame TCP target, in meters")
    parser.add_argument("--backend", choices=("reduced", "full"), help="Override QP backend")
    parser.add_argument("--output", type=Path, default=Path("results/latest"),
                        help="Results directory; existing run files here are overwritten")
    parser.add_argument("--ros", action="store_true", help="Enable the optional localhost ROS bridge and pacing")
    parser.add_argument("--quiet", action="store_true", help="Suppress periodic progress messages")
    options = parser.parse_args(argv)
    if options.hz is not None and (not math.isfinite(options.hz) or options.hz <= 0):
        parser.error("--hz must be finite and positive")
    if options.steps is not None and options.steps < 1:
        parser.error("--steps must be positive")
    return options


def configure_numerics():
    """Set thread counts before importing NumPy/JAX or creating the simulator."""
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def load_run_config(options):
    from xarm_density.config import load_config, validate_config

    config = load_config(options.config)
    if options.goal is not None:
        config["goal"] = options.goal
    if options.hz is not None:
        config["dt_s"] = 1 / options.hz
    if options.backend is not None:
        config["qp_backend"] = options.backend
    if options.steps is not None:
        config["duration_s"] = options.steps * config["dt_s"]
    config = validate_config(config)
    steps = round(config["duration_s"] / config["dt_s"])
    config["duration_s"] = steps * config["dt_s"]
    return config


def print_run_settings(config, options):
    from xarm_density.config import DEFAULT_CONFIG

    mode = "headless" if options.headless else "PyBullet GUI"
    print(f"UF850 CDF-QP | {mode} | simulation only", flush=True)
    print(f"Config: {(options.config or DEFAULT_CONFIG).resolve()}", flush=True)
    print(f"Rate: {1/config['dt_s']:g} Hz ({1000*config['dt_s']:g} ms) | "
          f"Duration: {config['duration_s']:g} s | QP: {config['qp_backend']}", flush=True)
    print(f"Start joints (rad): {config['q_start']}", flush=True)
    print(f"TCP goal (m): {config['goal']} | Obstacles: {len(config['obstacle_radii'])}", flush=True)
    print("Loading robot and compiling controller; initial startup may take a few seconds...", flush=True)


@contextmanager
def simulation_session(config, options):
    from xarm_density.simulation import Simulation

    simulation = bridge = None
    try:
        simulation = Simulation(config, gui=not options.headless)
        if options.ros:
            from xarm_density.bridge import SimulationBridge
            bridge = SimulationBridge()
        yield simulation, bridge
    finally:
        if bridge is not None:
            bridge.close()
        if simulation is not None:
            simulation.close()


def apply_ros_target(simulation, bridge):
    if bridge is None:
        return
    goal = bridge.receive_goal()
    if goal is not None:
        try:
            simulation.set_goal(goal)
        except ValueError as error:
            print(f"Rejected ROS target: {error}", flush=True)


def run_control_loop(simulation, config, options, bridge=None):
    """Advance the same controller used by both launchers; diagnostics are in step()."""
    steps = round(config["duration_s"] / simulation.dt)
    paced = options.ros or (not options.headless and not options.fast)
    progress_interval = max(1, round(1 / simulation.dt))
    for index in range(steps):
        start = perf_counter()
        apply_ros_target(simulation, bridge)
        record = simulation.step()
        if bridge is not None:
            bridge.publish(simulation, record)
        if not options.quiet and ((index + 1) % progress_interval == 0 or index + 1 == steps):
            print(f"  {index+1}/{steps} steps | t={record['time_s']:.2f} s | "
                  f"target error={1000*record['target_error_m']:.1f} mm | "
                  f"sphere clearance={1000*record['sphere_clearance_m']:.1f} mm", flush=True)
        if paced:
            sleep(max(0, simulation.dt - (perf_counter() - start)))


def save_results(simulation, config, output, status="completed"):
    """Save full or partial evidence; CSV column names stay backwards compatible."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = simulation.summary()
    summary["run_status"] = status
    summary["requested_steps"] = round(config["duration_s"] / config["dt_s"])
    documents = {"metrics.json": summary, "config.json": config, "scene.json": config}
    for name, document in documents.items():
        (output / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    columns = list(SCALAR_COLUMNS) + [f"{key}_{i}" for key, size in VECTOR_COLUMNS.items()
                                     for i in range(size)]
    with (output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in simulation.records:
            row = {key: record[key] for key in SCALAR_COLUMNS}
            for key, size in VECTOR_COLUMNS.items():
                row.update({f"{key}_{i}": record[key][i] for i in range(size)})
            writer.writerow(row)
    return summary


def print_summary(summary, output):
    reached = "reached" if summary["target_reached"] else "not reached within this run"
    print(f"Finished: {summary['steps']} steps | target {reached} | "
          f"final error={1000*summary['final_target_error_m']:.2f} mm", flush=True)
    print(f"Control time: median={summary['control_median_ms']:.3f} ms | "
          f"p95={summary['control_p95_ms']:.3f} ms | max={summary['control_max_ms']:.3f} ms | "
          f"deadline misses={summary['control_deadline_misses']}", flush=True)
    print(f"Obstacle-contact steps: {summary['contact_steps']}", flush=True)
    print(f"Results: {Path(output).resolve()} (config.json, metrics.json, trajectory.csv)", flush=True)


def run_experiment(config, options):
    """Installed-command workflow, using the same helpers as the root example."""
    print_run_settings(config, options)
    with simulation_session(config, options) as (simulation, bridge):
        try:
            run_control_loop(simulation, config, options, bridge)
        except (KeyboardInterrupt, ValueError, RuntimeError, OSError):
            save_results(simulation, config, options.output, status="stopped")
            raise
        summary = save_results(simulation, config, options.output)
        print_summary(summary, options.output)


def run_cli(argv=None, workflow=run_experiment):
    options = parse_arguments(argv)
    configure_numerics()
    try:
        config = load_run_config(options)
        workflow(config, options)
        return 0
    except KeyboardInterrupt:
        print(f"Stopped by user. Partial results, if initialized: {options.output.resolve()}", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Simulation stopped: {error}", file=sys.stderr)
        return 1


def main(argv=None):
    return run_cli(argv)
