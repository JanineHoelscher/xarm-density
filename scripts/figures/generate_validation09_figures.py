#!/usr/bin/env python3
"""Create paper figures from saved validation 9 data, without rerunning control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


BLUE, ORANGE, GREEN, RED = "#0072B2", "#D55E00", "#009E73", "#CC3311"
SINGLE, DOUBLE = 3.5, 7.16


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_model():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def save_figure(fig, folder, name):
    fig.align_ylabels()
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=600, bbox_inches="tight",
                    pad_inches=0.035, facecolor="white")
    plt.close(fig)


def axes_style(axes, duration):
    for ax in np.asarray(axes).ravel():
        ax.set_xlim(0, duration)
        ax.grid(True, color="0.90", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(4))


def export_table(folder, name, rows):
    with (folder / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", "Value", "Unit / interpretation"])
        writer.writerows(rows)
    def escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{tabular}{lrl}",
             r"\toprule", r"Metric & Value & Unit / interpretation \\", r"\midrule"]
    lines.extend(" & ".join(escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (folder / f"{name}.tex").write_text("\n".join(lines) + "\n")


def md_table(rows):
    return "\n".join(["| Metric | Value | Unit / interpretation |",
                      "|---|---:|---|"] +
                     [f"| {a} | {b} | {c} |" for a, b, c in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = args.input or here.parent.parent / "results/cdf_qp_validations/validation_09"
    folder = args.output or here
    folder.mkdir(parents=True, exist_ok=True)
    archive_path, metrics_path = source / "trajectory.npz", source / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    config, recorded = payload["config"], payload["metrics"]
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = len(data["filtered_controls"])
    dt = float(config["dt"])
    state_time = np.arange(n + 1) * dt
    control_time = state_time[:-1]
    clearance_time = state_time[1:]
    duration = n * dt
    state_keys = ("q_states", "ee_jax_states", "ee_bullet_states",
                  "reference_positions", "orientation_errors")
    for key, values in data.items():
        expected = n + 1 if key in state_keys else n
        if len(values) != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid samples in {key}")
    actual, reference = data["ee_jax_states"], data["reference_positions"]
    position_error = np.linalg.norm(actual - reference, axis=1)
    orientation_error = np.rad2deg(data["orientation_errors"])
    nominal, filtered = data["nominal_controls"], data["filtered_controls"]
    intervention = np.linalg.norm(filtered - nominal, axis=1)
    qp_ms, solver_ms = data["solve_times"] * 1e3, data["solver_times"] * 1e3
    margin = float(config["residual_margin"])
    residual_tolerance = 1e-8
    stats = {
        "position_rms_mm": rms(position_error) * 1e3,
        "position_max_mm": float(position_error.max()) * 1e3,
        "position_final_mm": float(position_error[-1]) * 1e3,
        "orientation_rms_deg": rms(orientation_error),
        "orientation_max_deg": float(orientation_error.max()),
        "orientation_final_deg": float(orientation_error[-1]),
        "intervention_rms_rad_s": rms(intervention),
        "intervention_max_rad_s": float(intervention.max()),
        "intervention_integral_rad": float(intervention.sum() * dt),
        "command_modified_percent": float(np.mean(intervention > 1e-6) * 100),
        "sphere_min_mm": float(data["sphere_clearances"].min() * 1e3),
        "mesh_min_mm": float(data["bullet_clearances"].min() * 1e3),
        "residual_min": float(data["residuals"].min()),
        "residual_margin_violation_steps_tol_1e8": int(np.sum(data["residuals"] < margin - residual_tolerance)),
        "contact_steps": int(np.count_nonzero(data["contact_counts"])),
        "joint_speed_max_rad_s": float(np.abs(filtered).max()),
        "fk_disagreement_max_m": float(np.linalg.norm(actual - data["ee_bullet_states"], axis=1).max()),
        "qp_mean_ms": float(qp_ms.mean()),
        "qp_median_ms": float(np.median(qp_ms)),
        "qp_p95_ms": float(np.percentile(qp_ms, 95)),
        "qp_max_ms": float(qp_ms.max()),
        "solver_median_ms": float(np.median(solver_ms)),
        "solver_p95_ms": float(np.percentile(solver_ms, 95)),
        "solver_max_ms": float(solver_ms.max()),
        "qp_exceeds_dt_percent": float(np.mean(qp_ms > dt * 1e3) * 100),
        "all_qps_feasible_recorded": bool(recorded["all_qps_feasible"]),
        "density_floor_activation": None,
    }
    checks = {
        "maximum_reference_tracking_error_m": position_error.max(),
        "final_target_error_m": np.linalg.norm(actual[-1] - config["goal"]),
        "maximum_orientation_error_deg": orientation_error.max(),
        "minimum_link_sphere_clearance_m": data["sphere_clearances"].min(),
        "minimum_pybullet_mesh_clearance_m": data["bullet_clearances"].min(),
        "mean_qp_time_ms": qp_ms.mean(),
    }
    for key, value in checks.items():
        if not np.isclose(value, recorded[key], rtol=1e-8, atol=1e-10):
            raise ValueError(f"Saved metrics disagree with trajectory: {key}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 3.5), sharex=True, layout="constrained")
    for j, ax in enumerate(axes):
        ax.plot(state_time, reference[:, j], color="0.25", linestyle="--", label="Reference")
        ax.plot(state_time, actual[:, j], color=BLUE, label="CDF-QP")
        ax.set_ylabel(f"${'xyz'[j]}$ (m)")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-1].set_xlabel("Time (s)")
    axes_style(axes, duration)
    save_figure(fig, folder, "fig01_tcp_tracking")

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 2.8), sharex=True, layout="constrained")
    axes[0].plot(state_time, position_error * 1e3, color=BLUE)
    axes[0].set_ylabel("Position error (mm)")
    axes[1].plot(state_time, orientation_error, color=ORANGE)
    axes[1].set_ylabel("Orientation error (deg)")
    axes[-1].set_xlabel("Time (s)")
    axes_style(axes, duration)
    save_figure(fig, folder, "fig02_tracking_errors")

    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE, 4.0), sharex=True, sharey=True, layout="constrained")
    for j, ax in enumerate(axes.ravel()):
        ax.axhline(config["qdot_limit"], color="0.65", linestyle=":", linewidth=0.7)
        ax.axhline(-config["qdot_limit"], color="0.65", linestyle=":", linewidth=0.7)
        ax.plot(control_time, nominal[:, j], color="0.25", linestyle="--", label="Nominal")
        ax.plot(control_time, filtered[:, j], color=BLUE, label="Filtered")
        ax.set_title(f"Joint {j + 1}", fontsize=8, loc="left", pad=3)
        ax.set_ylabel(r"$\dot q$ (rad/s)")
        ax.set_ylim(-config["qdot_limit"] * 1.15, config["qdot_limit"] * 1.15)
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes_style(axes, duration)
    save_figure(fig, folder, "fig03_joint_velocities")

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.0), sharex=True, layout="constrained")
    axes[0].plot(clearance_time, data["sphere_clearances"] * 1e3, color=BLUE, label="Link spheres")
    axes[0].plot(clearance_time, data["bullet_clearances"] * 1e3, color=ORANGE, linestyle="--", label="Bullet mesh")
    axes[0].axhline(0, color=RED, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Clearance (mm)")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[1].plot(control_time, data["residuals"], color=GREEN, label="CDF residual")
    axes[1].axhline(margin, color="0.3", linestyle="--", linewidth=0.8, label=r"Margin $\eta$")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    lo, hi = min(data["residuals"].min(), margin), max(data["residuals"].max(), margin)
    pad = max((hi - lo) * .15, abs(margin) * .1, 1e-5)
    axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].set_ylabel("CDF residual")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    axes[2].plot(control_time, intervention, color=BLUE)
    axes[2].set_ylabel("Intervention (rad/s)")
    axes[2].set_xlabel("Time (s)")
    axes_style(axes, duration)
    save_figure(fig, folder, "fig04_safety_intervention")

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.2), layout="constrained")
    axes[0].plot(control_time, qp_ms, color=BLUE, linewidth=.6, label="QP assembly + solve")
    axes[0].plot(control_time, solver_ms, color=ORANGE, linewidth=.6, alpha=.9, label="Solver")
    axes[0].axhline(dt * 1e3, color="0.25", linestyle=":", label="Simulation step")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_xlabel("Simulated time (s)")
    axes[0].set_xlim(0, duration)
    axes[0].legend(frameon=False, fontsize=6, ncol=2,
                   loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    for values, color, label, style in [(qp_ms, BLUE, "QP assembly + solve", "-"),
                                         (solver_ms, ORANGE, "Solver", "--")]:
        axes[1].step(np.sort(values), np.arange(1, n + 1) / n,
                     where="post", color=color, linestyle=style, label=label)
    axes[1].axvline(dt * 1e3, color="0.25", linestyle=":", linewidth=.8)
    axes[1].set_xlabel("Recorded time (ms)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="0.9", linewidth=.4)
    save_figure(fig, folder, "fig05_recorded_timing")

    f = lambda key: f"{stats[key]:.3f}"
    performance_rows = [
        ("Position error RMS", f("position_rms_mm"), "mm"),
        ("Position error maximum", f("position_max_mm"), "mm"),
        ("Position error final", f("position_final_mm"), "mm"),
        ("Orientation error RMS", f("orientation_rms_deg"), "deg"),
        ("Orientation error maximum", f("orientation_max_deg"), "deg"),
        ("Orientation error final", f("orientation_final_deg"), "deg"),
        ("Minimum sphere clearance", f("sphere_min_mm"), "mm"),
        ("Minimum mesh clearance", f("mesh_min_mm"), "mm"),
        ("Obstacle contact steps", stats["contact_steps"], "steps"),
        ("All QPs feasible", "yes" if stats["all_qps_feasible_recorded"] else "no", "saved summary"),
        ("Minimum CDF residual", f("residual_min"), "raw residual"),
        ("Maximum joint speed", f("joint_speed_max_rad_s"), "rad/s"),
        ("Intervention RMS", f("intervention_rms_rad_s"), "rad/s"),
        ("Intervention maximum", f("intervention_max_rad_s"), "rad/s"),
        ("Integrated intervention norm", f("intervention_integral_rad"), "rad"),
    ]
    timing_rows = [
        ("QP assembly + solve median", f("qp_median_ms"), "ms"),
        ("QP assembly + solve p95", f("qp_p95_ms"), "ms"),
        ("QP assembly + solve maximum", f("qp_max_ms"), "ms"),
        ("Solver median", f("solver_median_ms"), "ms"),
        ("Solver p95", f("solver_p95_ms"), "ms"),
        ("Solver maximum", f("solver_max_ms"), "ms"),
        ("Simulated timestep", f"{dt * 1e3:.3f}", "ms"),
        ("QP time above simulated timestep", f("qp_exceeds_dt_percent"), "% of recorded steps"),
    ]
    export_table(folder, "table01_performance", performance_rows)
    export_table(folder, "table02_timing", timing_rows)
    provenance = {
        "input_trajectory_sha256": digest(archive_path),
        "input_metrics_sha256": digest(metrics_path),
        "simulation_rerun": False,
        "samples": {"states": n + 1, "controls": n},
        "dt_s": dt, "duration_s": duration,
        "figure_generation_environment_only": {
            "cpu": cpu_model(), "platform": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "original_timing_processor_and_backend": "not recorded in source artifacts",
        "statistics": stats,
    }
    (folder / "summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    readme = f"""# Validation 9: paper figures and evidence

These figures use only the latest saved validation 9 trajectory and metrics.
The simulator was not rerun. This is one closed-gripper UF850 trajectory through
five static spherical obstacles, using the whole-arm CDF-QP and a pose-tracking
nominal controller. The TCP follows a seven-second minimum-jerk reference;
the orientation reference is the initial tool orientation.

## Reproduce

```bash
conda activate xarm
cd /home/sriramk/xarm_ws
python scripts/figures/generate_validation09_figures.py
```

Optional `--input PATH` selects a validation 9 result directory, and `--output
PATH` selects the output directory. Defaults resolve relative to this script.
Requires NumPy and Matplotlib only; no JAX, GPU, ROS, or simulation is needed.
Regeneration overwrites this folder's generated figures, tables, and README.

Inputs: `results/cdf_qp_validations/validation_09/trajectory.npz` and
`metrics.json`. Their SHA-256 hashes, sample counts, and computed statistics
are in [summary.json](summary.json). State histories have {n + 1} samples;
command/residual/clearance histories have {n}. Duration is {duration:.3f} s.

## Figure style and paper use

PDFs are vector exports with embedded TrueType fonts; PNG previews are 600 dpi.
Plots use serif fonts, 7--8 pt labels, restrained grids, and contrasting colors
with line styles for paired signals. Single-column designs target 3.5 inches;
the six-joint comparison targets 7.16 inches across two columns. Tight cropping
can slightly change exported page size; set the width when including a PDF.
Check final readability in your conference template rather than shrinking a
two-column figure into one column. Colors also distinguish related signals;
captions and dash patterns preserve paired-curve identification in grayscale.

```latex
% In the preamble:
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
% In the paper (paths assume the repository is available):
\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=\\columnwidth]{{scripts/figures/fig02_tracking_errors.pdf}}
  \\caption{{TCP position and orientation tracking errors for the UF850
  whole-arm CDF-QP in validation 9.}}
  \\label{{fig:uf850-tracking}}
\\end{{figure}}
```

Tables are provided as CSV and booktabs LaTeX fragments. Add a table environment,
caption, and label in your paper; the fragments are not standalone documents.
Some tables are wide and should use a two-column table or a selected subset
of rows, with ordinary font size instead of scaling the full table down.

## Tracking performance

![TCP reference and actual position](fig01_tcp_tracking.png)

[Vector PDF](fig01_tcp_tracking.pdf). Reference and actual TCP coordinates
share identical timestamps. The position shown is the JAX FK TCP; the maximum
saved JAX/Bullet FK disagreement is {stats['fk_disagreement_max_m']:.3e} m.

![Tracking errors](fig02_tracking_errors.png)

[Vector PDF](fig02_tracking_errors.pdf). Position error is the Euclidean
distance to the moving reference, not the final goal. Orientation error is
the SO(3) rotation angle relative to the fixed initial orientation. RMS is
the square root of the arithmetic mean of squared scalar errors over all
{n + 1} uniformly spaced state samples, including the initial state.

![Nominal and filtered joint velocities](fig03_joint_velocities.png)

[Vector PDF](fig03_joint_velocities.pdf). Dashed curves are nominal commands;
solid curves are filtered commands. Dotted lines mark the joint speed limits.
Commands at time t act over the following integration interval.

## Clearance, residual, and intervention

![Safety and controller intervention](fig04_safety_intervention.png)

[Vector PDF](fig04_safety_intervention.pdf). Clearances are the minimum across
modeled sphere-obstacle pairs and independent Bullet mesh-obstacle queries.
They describe different geometry and are not expected to coincide. Zero
clearance is the collision threshold; these are obstacle contacts, not a
self-collision validation. Clearance is logged after each step at t+dt,
while residual and input intervention are evaluated at t.

Intervention is the joint-vector norm `||u_filtered - u_nominal||_2`.
Its RMS and maximum quantify command modification, not a causal comparison
against an unfiltered trajectory. Integrated intervention uses a zero-order
hold sum `dt * sum(intervention)`. Command modification above 1e-6 rad/s occurs
on {stats['command_modified_percent']:.2f}% of steps; regularization can also
modify commands, so this is not a safety-constraint activation percentage.

The raw residual is compared with eta = {margin:g}, using a tolerance of
1e-8 for counting numerical margin violations. This trace has
{stats['residual_margin_violation_steps_tol_1e8']} such violations. A positive
pointwise residual alone is not a sampled-data invariance proof.

### Table 1: tracking and safety

{md_table(performance_rows)}

[CSV](table01_performance.csv) | [LaTeX](table01_performance.tex)

Target reached: **{'yes' if recorded['target_reached'] else 'no'}**, according
to the configured {config['target_radius'] * 1e3:.1f} mm goal tolerance.
Feasibility is taken from the saved metrics summary: per-step status strings
are not present in the NPZ file. Density and density-floor activation were
not logged, so their activation counts cannot be reported from this archive.

## Recorded QP timing

![Recorded QP times and empirical distributions](fig05_recorded_timing.png)

[Vector PDF](fig05_recorded_timing.pdf). These are saved per-step measurements
after the runner's explicit nominal/CDF warm-up, not fresh benchmarks. The
upper plot uses simulated time; the lower plot is the empirical cumulative
distribution of recorded wall-clock durations. No smoothing or sample dropping
is used. The QP timer covers constraint/sparse-matrix assembly, Clarabel setup
or update, and solve. It excludes earlier JAX evaluation, objective/Hessian
assembly and the eigenvalue check, nominal control, Bullet checks, rendering,
and recording. Solver time is the narrower timed solver call, including any
retry. Neither is full controller latency.

### Table 2: recorded timing

{md_table(timing_rows)}

[CSV](table02_timing.csv) | [LaTeX](table02_timing.tex)

The original run did not record processor, JAX backend, thread count, or package
versions. Current figure-generation host: **{cpu_model()}**; this does not
establish the hardware used for the saved timing. QP time exceeds the simulated
timestep on {stats['qp_exceeds_dt_percent']:.2f}% of steps, so this archive
does not demonstrate 240 Hz real-time control. Report simulation duration and
wall-clock computation separately.

## Requested evidence and remaining work

1. **Geometry comparison.** Provide synchronized trajectories for validations
   5 and 6, minimum whole-arm sphere/mesh clearance, contacts, target error,
   and intervention; verify safety geometry is the only changed setting.
   **Pending:** intentionally not generated here because this analysis uses
   validation 9 only. A settings audit is required before making that claim.
2. **Tracking performance.** Provide validation 9 position/orientation errors,
   reference/actual TCP histories, nominal/filtered velocities, residual, both
   clearances, and RMS/maximum errors. **Completed for the saved run** in
   figures 1--4 and table 1.
3. **Matched baseline.** Run CDF-QP and OSCBF with identical nominal control,
   geometry, scenes, limits, and timestep; compare tracking, clearance,
   contacts, feasibility, intervention, and computation. **Pending:** no matched
   OSCBF trajectory is inferred or synthesized from this single CDF-QP run.
4. **Repeatability and timing.** Repeat representative starts and obstacle
   placements; record success, collisions, infeasibility, density-floor use,
   and post-compilation median/p95/maximum full-controller latency plus
   processor details. **Pending:** table 2 reports the available single-run
   QP timing only. Add density-floor counters, full-cycle timers, environment
   metadata, and trial identifiers before running the study.

Validation 10's [TODO](../../results/cdf_qp_validations/validation_10/TODO.md)
covers interactive simulation and CPU/GPU profiling. This figure set does not
establish robustness over trials, hardware performance, motor dynamics, or
a CDF-QP advantage over a matched OSCBF baseline.
"""
    (folder / "README.md").write_text(readme)
    print(json.dumps(stats, indent=2))
    print(f"Saved five PDF/PNG figures, two CSV/LaTeX tables, summary, and README to {folder}")


if __name__ == "__main__":
    main()
