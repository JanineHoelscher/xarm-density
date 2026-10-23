"""One standalone simulation entry point, with numerical settings set early."""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser(description="UF850 whole-arm CDF-QP simulation")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Disable wall-clock pacing")
    parser.add_argument("--hz", type=float, default=100.)
    parser.add_argument("--steps", type=int, help="Override the seven-second run length")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--goal", type=float, nargs=3, metavar=("X","Y","Z"))
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    parser.add_argument("--backend", choices=("reduced", "full"), default="reduced")
    args = parser.parse_args()
    if args.hz <= 0 or args.steps is not None and args.steps < 1:
        parser.error("Rate and step count must be positive")
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from xarm_density.simulation import Simulation, load_scene
    scene = load_scene(args.scene)
    if args.goal is not None:
        scene["goal"] = args.goal
    simulation = None
    try:
        print("Loading bundled robot and compiling controller...", flush=True)
        simulation = Simulation(scene, args.hz, gui=not args.headless, backend=args.backend)
        steps = args.steps or round(scene["reference_duration_s"]*args.hz)
        print(f"Running {steps} steps at {args.hz:g} Hz; backend={args.backend}", flush=True)
        for _ in range(steps):
            start = perf_counter()
            simulation.step()
            if not args.headless and not args.fast:
                sleep(max(0, simulation.dt-(perf_counter()-start)))
        summary = simulation.summary()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
        (args.output / "scene.json").write_text(json.dumps(scene, indent=2)+"\n")
        fields = ["time_s", "target_error_m", "tracking_error_m", "sphere_clearance_m",
                  "mesh_clearance_m", "contacts", "residual", "control_time_s", "solver_time_s", "loop_time_s"]
        columns = fields + [f"{key}_{j}" for key in ("q","command","nominal","tcp","reference")
                            for j in range(6 if key in ("q","command","nominal") else 3)]
        with (args.output / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in simulation.records:
                row = {key: record[key] for key in fields}
                for key in ("q","command","nominal","tcp","reference"):
                    row.update({f"{key}_{j}": value for j,value in enumerate(record[key])})
                writer.writerow(row)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Saved scene, metrics, and raw CSV to {args.output}")
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Simulation stopped: {error}\n")
    finally:
        if simulation is not None:
            simulation.close()
