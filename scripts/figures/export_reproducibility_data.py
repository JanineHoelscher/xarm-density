#!/usr/bin/env python3
"""Export raw figure data as round-trip-checked CSV, or restore plotting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(data_dir, destination):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for dataset in manifest["datasets"]:
        arrays = {}
        for group in dataset["groups"]:
            path = data_dir / group["csv"]
            if sha(path) != group["sha256"]:
                raise ValueError(f"CSV checksum mismatch: {path}")
            matrix = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            for spec in group["arrays"]:
                values = matrix[:, spec["start"]:spec["stop"]]
                arrays[spec["key"]] = values.astype(spec["dtype"]).reshape(spec["shape"])
        path = destination / dataset["source"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    for resource in manifest["json_inputs"]:
        source = data_dir / resource["copy"]
        if sha(source) != resource["sha256"]:
            raise ValueError(f"JSON checksum mismatch: {source}")
        target = destination / resource["source"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def export(root, folder):
    folder.mkdir(parents=True, exist_ok=True)
    base = "results/cdf_qp_validations/"
    sources = [(base + "validation_09/trajectory.npz", "validation09", 1 / 240),
               (base + "validation_10/performance/replay_timing.npz", "validation10_replay", None)]
    benchmark = json.loads((root / (base + "validation_10/performance/benchmark.json")).read_text())
    for row in benchmark["rollouts"]:
        rate = row["rate_hz"]
        sources.append((base + f"validation_10/performance/rollout_{rate:g}hz.npz",
                        f"validation10_{rate:g}hz", 1 / rate))
    manifest = {"format_version": 1, "float_encoding": "17 significant decimal digits",
                "datasets": [], "json_inputs": [], "benchmark_environment": benchmark["environment"]}
    for relative, stem, dt in sources:
        path = root / relative
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        groups = {}
        for key, values in arrays.items():
            groups.setdefault(len(values), []).append(key)
        dataset = {"source": relative, "source_sha256": sha(path), "dt_s": dt, "groups": []}
        for count, keys in groups.items():
            columns = ["sample_index"]
            pieces = [np.arange(count)[:, None]]
            if dt is not None:
                columns.append("time_s")
                pieces.append((np.arange(count) * dt)[:, None])
                if "q_states" not in keys and "ee_jax_states" not in keys:
                    columns.append("next_time_s")
                    pieces.append(((np.arange(count) + 1) * dt)[:, None])
            elif count != len(arrays["indices"]):
                indices = arrays["indices"]
                columns.extend(["repeat_index", "saved_state_index", "saved_state_time_s"])
                pieces.extend([(np.arange(count) // len(indices))[:, None],
                               np.tile(indices, count // len(indices))[:, None],
                               (np.tile(indices, count // len(indices)) / 240)[:, None]])
            specs = []
            offset = len(columns)
            for key in keys:
                values = arrays[key]
                flat = values.reshape(count, -1)
                width = flat.shape[1]
                columns.extend([key] if width == 1 else [f"{key}_{j}" for j in range(width)])
                specs.append({"key": key, "shape": list(values.shape), "dtype": str(values.dtype),
                              "start": offset, "stop": offset + width})
                pieces.append(flat)
                offset += width
            target = folder / f"{stem}_{count}_samples.csv"
            np.savetxt(target, np.column_stack(pieces), delimiter=",", header=",".join(columns),
                       comments="", fmt="%.17g")
            loaded = np.loadtxt(target, delimiter=",", skiprows=1, ndmin=2)
            for spec in specs:
                recovered = loaded[:, spec["start"]:spec["stop"]].astype(spec["dtype"]).reshape(spec["shape"])
                if not np.array_equal(recovered, arrays[spec["key"]], equal_nan=True):
                    raise ValueError(f"CSV round-trip failed: {target}: {spec['key']}")
            dataset["groups"].append({"csv": target.name, "sha256": sha(target),
                                       "columns": columns, "arrays": specs})
        manifest["datasets"].append(dataset)
    for relative, name in [(base + "validation_09/metrics.json", "validation09_metadata.json"),
                           (base + "validation_10/performance/benchmark.json", "validation10_metadata.json")]:
        source, target = root / relative, folder / name
        target.write_bytes(source.read_bytes())
        manifest["json_inputs"].append({"source": relative, "copy": name, "sha256": sha(target)})
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def document(folder, manifest):
    rows = []
    for dataset in manifest["datasets"]:
        for group in dataset["groups"]:
            rows.append(f"| [{group['csv']}]({group['csv']}) | {', '.join(s['key'] for s in group['arrays'])} |")
    text = """# Raw CSV data and reproduction

These are raw samples, not just rounded paper-table summaries. CSV floats use
17 significant digits and are checked for exact numerical round-trip against
every source NPZ array. Source file hashes, exported CSV hashes, column slices,
shapes, and dtypes are recorded in [manifest.json](manifest.json).
Saved JSON metadata preserves the scene, gains, timestep, benchmark settings,
and recorded environment. ZIP metadata in reconstructed NPZ files may differ;
array contents and dtypes are preserved.

## Files

| CSV | Arrays |
|---|---|
""" + "\n".join(rows) + """

Replay CSVs retain all source arrays for audit/reproduction; only the optimized
timing columns are plotted in the paper figures. Saved-state indices and repeat
indices identify each timed evaluation. No samples are filtered or smoothed.

## Units and timestamps

- `q_states`: radians; `nominal_controls`, `filtered_controls`, `speed`: rad/s.
- `ee_*`, `reference_positions`, `*_clearance*`, `position_error_m`: meters.
- `orientation_errors`: radians; paper plots convert to degrees.
- `solve_times`, `solver_times`, replay `*_time`, and rollout `*_time_s`: seconds;
  plots/tables convert these to milliseconds.
- `residuals`/`residual`: raw CDF residual, compared with eta = 0.01.
- `contact_counts`/`contacts`: contact count; `floor`: density-floor flag (0/1).
- State-history `time_s` is the state timestamp, including the initial state.
- Step-history `time_s` is the command/residual timestamp. Clearance, contacts,
  and tracking diagnostics in that row refer to `next_time_s`, after integration.
- Replay `saved_state_time_s` refers to the source 240 Hz trajectory, not elapsed
  benchmark wall time. Timing values in the same row are measured wall durations.
- Extra index/time columns are export annotations, not original NPZ arrays.

## How the results were obtained

1. Validation 9 supplies the saved TCP reference/actual position, orientation
   error, commands, residual, obstacle clearance, and recorded solver timings.
2. Validation 10 profiles 160 saved states over three repetitions after explicit
   warm-up, using the seven-coordinate CDF-QP with fused JAX evaluation. No
   safety geometry, objective weights, or command limits are relaxed.
3. Fresh headless kinematic rollouts check 100, 50, and 240 Hz integration.
   Control timing includes reference generation, nominal controller, and density
   filter. Diagnostic-loop timing additionally includes Bullet integration and
   collision/clearance/tracking queries; neither includes GUI rendering.
4. Plotters calculate median, p95 (NumPy linear-interpolated percentile), and
   maximum from every saved measured sample after the runner's warm-up.
   Position error is the reference-to-actual TCP Euclidean distance. Orientation
   error is the rotation angle relative to the fixed orientation reference.
   RMS is sqrt(mean(error**2)); validation 9 includes its initial state, whereas
   validation 10 rollout RMS uses its logged post-step samples.
5. Minimum clearance is the minimum of the saved per-step whole-arm clearance.
   Contacts and deadline misses count affected steps. Timing budget use is
   `100 * measured_time / timestep`. All unit conversions occur only for display.

The nominal-plus-filter, filter-only, and solver-only timers are nested and
must not be added. Benchmark processor/device/version details are saved in
validation10_metadata.json. The original validation 9 archive does not include
that original run's processor/backend information. These measurements do not
establish worst-case real-time guarantees or robustness across different scenes.

## Reproduce figures from CSV alone

```bash
conda activate xarm
cd /home/sriramk/xarm_ws
python scripts/figures/generate_figures.py --from-csv
```

This validates checksums, restores temporary plotting inputs from these CSVs
and JSONs, and recreates the PDFs, PNGs, summary tables, and figure README.
It does not rerun the simulator or need the original NPZ archives. Temporary
ZIP files can have different binary hashes while holding identical arrays.

To export a newer set of saved runs and regenerate everything:

```bash
python scripts/figures/generate_figures.py
```

To collect new timing measurements, rather than replot the saved ones:

```bash
python scripts/pybullet_validation_10.py --rates 100 50 240 --samples 160 --repeats 3 --blas-threads 1 --device cpu
python scripts/figures/generate_figures.py
```

New measurements can vary with OS load, CPU clocks, and software versions.
The required local robot/dependency setup is described in the workspace README.
Raw timing captures, not a promise of identical future timing, are reproduced here.
"""
    (folder / "README.md").write_text(text)
    parent = folder.parent / "README.md"
    if parent.exists():
        content = parent.read_text().split("<!-- RAW_CSV_REPRODUCTION -->", 1)[0].rstrip()
        content += """

<!-- RAW_CSV_REPRODUCTION -->

## Raw CSVs and exact figure reproduction

Raw state, command, residual, clearance, and timing samples are saved under
[data/](data/README.md), alongside scene/environment JSONs, units, timestamp
definitions, calculation methods, and SHA-256 checksums. This complements the
rounded summary-table CSVs above. The export checks exact numeric round-trip.

```bash
python scripts/figures/generate_figures.py --from-csv
```

This rebuilds the figures using the raw CSVs and JSON metadata alone. For a new
benchmark use the collection command in [data/README.md](data/README.md).
"""
        parent.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-to", type=Path)
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    folder = here / "data"
    if args.restore_to:
        restore(folder, args.restore_to)
        print(f"Restored CSV inputs to {args.restore_to}")
        return
    manifest = (json.loads((folder / "manifest.json").read_text()) if args.describe_only
                else export(here.parent.parent, folder))
    document(folder, manifest)
    print("Raw CSVs, checksums, methods, and reproduction instructions ready")


if __name__ == "__main__":
    main()
