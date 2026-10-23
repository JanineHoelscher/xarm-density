#!/usr/bin/env python3
"""Add validation 10 performance figures to the validation 9 paper figure set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_validation09_figures import BLUE, ORANGE, GREEN, DOUBLE, digest, save_figure


def write_table(folder, stem, headers, rows):
    with (folder / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    def escaped(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")
    lines = [r"% Requires \usepackage{booktabs}",
             r"\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
             r"\toprule", " & ".join(map(escaped, headers)) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(escaped, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (folder / f"{stem}.tex").write_text("\n".join(lines) + "\n")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] + ["---:"] * (len(headers)-1)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    folder = args.output or here
    source = args.input or here.parent.parent / "results/cdf_qp_validations/validation_10/performance"
    folder.mkdir(parents=True, exist_ok=True)
    report_path, timing_path = source / "benchmark.json", source / "replay_timing.npz"
    report = json.loads(report_path.read_text())
    with np.load(timing_path, allow_pickle=False) as archive:
        replay = {name: archive[name] for name in archive.files}
    counts = []
    for backend in ("full", "reduced"):
        for component in ("total_time", "nominal_time", "controller_time", "solver_solve_time"):
            ms = replay[f"{backend}_{component}"] * 1000
            assert np.all(np.isfinite(ms)) and np.all(ms > 0)
            counts.append(len(ms))
            for statistic, computed in [("median_ms", np.median(ms)),
                                         ("p95_ms", np.percentile(ms, 95)),
                                         ("max_ms", ms.max())]:
                np.testing.assert_allclose(computed, report[backend][component][statistic], rtol=1e-10)
    assert len(set(counts)) == 1
    rollouts = sorted(report["rollouts"], key=lambda row: row["rate_hz"])
    hashes = {report_path.name: digest(report_path), timing_path.name: digest(timing_path)}
    for row in rollouts:
        rate = row["rate_hz"]
        path = source / f"rollout_{rate:g}hz.npz"
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            for field, key in [("control_time_s", "control"),
                               ("loop_time_s", "loop_with_diagnostics")]:
                values = data[field] * 1000
                for statistic, computed in [("median_ms", np.median(values)),
                                             ("p95_ms", np.percentile(values, 95)),
                                             ("max_ms", values.max())]:
                    np.testing.assert_allclose(computed, row[key][statistic], rtol=1e-10)
            np.testing.assert_allclose(data["mesh_clearance_m"].min(), row["minimum_mesh_clearance_m"])
            np.testing.assert_allclose(data["sphere_clearance_m"].min(), row["minimum_sphere_clearance_m"])
            assert np.count_nonzero(data["contacts"]) == row["contact_steps"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": .6, "lines.linewidth": 1.1,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), layout="constrained")
    components = [("total_time", "Nominal + filter", BLUE, "-"),
                  ("controller_time", "Density filter", GREEN, "--"),
                  ("solver_solve_time", "Solver only", ORANGE, ":")]
    for component, label, color, style in components:
        ms = np.sort(replay[f"reduced_{component}"] * 1000)
        axes[0].step(ms, np.arange(1, len(ms) + 1) / len(ms), where="post",
                     label=label, color=color, linestyle=style)
    axes[0].set_xscale("log")
    axes[0].set_xticks([.05, .1, .5, 1, 2], ["0.05", "0.1", "0.5", "1", "2"])
    axes[0].set_xlabel("Measured time (ms)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False, loc="center")
    x = np.arange(3)
    for offset, (component, label, color, _) in zip((-.24, 0, .24), components):
        values = [report["reduced"][component][key] for key in ("median_ms", "p95_ms", "max_ms")]
        axes[1].bar(x + offset, values, width=.22, color=color,
                    edgecolor="white", linewidth=.5, label=label)
        for position, value in zip(x + offset, values):
            axes[1].text(position, value + .025, f"{value:.2f}", ha="center", va="bottom", fontsize=6)
    axes[1].set_xticks(x, ["Median", "95th percentile", "Maximum"])
    axes[1].set_ylabel("Measured time (ms)")
    axes[1].set_ylim(0, report["reduced"]["total_time"]["max_ms"] * 1.3)
    axes[1].legend(frameon=False, ncol=2, loc="upper left", fontsize=6)
    for j, ax in enumerate(axes):
        ax.set_title(f"({'ab'[j]}) " + ["Optimized timing distributions", "Optimized timing statistics"][j],
                     fontsize=8, loc="left", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", linewidth=.4)
        ax.set_axisbelow(True)
    save_figure(fig, folder, "fig06_optimized_timing")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True, layout="constrained")
    for j, (ax, component) in enumerate(zip(axes, ("control", "loop_with_diagnostics"))):
        for offset, key, label, color, hatch in [(-.24, "median_ms", "Median", GREEN, ""),
                                                 (0, "p95_ms", "95th percentile", BLUE, "//"),
                                                 (.24, "max_ms", "Maximum", ORANGE, "..")]:
            values = [100 * row[component][key] / row["dt_ms"] for row in rollouts]
            ax.bar(np.arange(len(rollouts)) + offset, values, width=.22, color=color,
                   hatch=hatch, edgecolor="white", linewidth=.5, label=label)
        ax.axhline(100, color=".3", linestyle="--", linewidth=.8)
        ax.set_xticks(np.arange(len(rollouts)), [f"{r['rate_hz']:g} Hz\n{r['dt_ms']:.2f} ms" for r in rollouts])
        ax.set_xlabel("Fixed simulation rate and step budget")
        ax.set_title(f"({'ab'[j]}) " + ["Nominal controller + filter", "Loop including Bullet diagnostics"][j],
                     fontsize=8, loc="left", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", linewidth=.4)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 115)
    axes[0].set_ylabel("Fraction of step budget (%)")
    axes[0].legend(frameon=False, ncol=3, fontsize=6, loc="upper left")
    axes[1].text(.98, 101, "Step budget", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color=".3")
    save_figure(fig, folder, "fig07_fixed_rate_budget")

    replay_rows = []
    for component, label, _, _ in components:
        replay_rows.append([label] + [f"{report['reduced'][component][k]:.3f}" for k in ("median_ms", "p95_ms", "max_ms")])
    replay_table = write_table(folder, "table03_optimized_timing",
                              ["Measured component", "Median (ms)", "p95 (ms)", "Max (ms)"], replay_rows)
    rate_rows = []
    safety_rows = []
    for r in rollouts:
        rate_rows.append([f"{r['rate_hz']:g}", f"{r['dt_ms']:.3f}",
                          f"{r['control']['p95_ms']:.3f}", f"{r['control']['max_ms']:.3f}",
                          f"{r['loop_with_diagnostics']['max_ms']:.3f}",
                          r["control_deadline_misses"], r["loop_deadline_misses"]])
        safety_rows.append([f"{r['rate_hz']:g}", f"{r['minimum_sphere_clearance_m']*1000:.2f}",
                            f"{r['minimum_mesh_clearance_m']*1000:.2f}",
                            f"{r['final_target_error_m']*1000:.2f}",
                            r["contact_steps"], r["density_floor_steps"],
                            "yes" if r["target_reached"] else "no"])
    rate_table = write_table(folder, "table04_fixed_rate_timing",
                            ["Hz", "Step (ms)", "Control p95", "Control max", "Loop max", "Control misses", "Loop misses"], rate_rows)
    safety_table = write_table(folder, "table05_fixed_rate_safety",
                              ["Hz", "Sphere (mm)", "Mesh (mm)", "Final error (mm)", "Contacts", "Floor count", "Reached"], safety_rows)
    (folder / "validation10_summary.json").write_text(json.dumps({
        "input_sha256": hashes, "replay_samples_per_backend": counts[0],
        "optimized_timing": report["reduced"], "environment": report["environment"],
        "equivalence": report["equivalence"], "rollouts": rollouts,
        "note": "Saved benchmark analysis only; no new timings or simulations collected.",
    }, indent=2) + "\n")
    readme_path = folder / "README.md"
    previous = readme_path.read_text()
    marker = "<!-- VALIDATION09_REPORT -->"
    if marker in previous:
        previous = previous.split(marker, 1)[1].lstrip()
    prefix = f"""# UF850 paper figures: tracking and optimized performance

This folder presents validation 9 tracking/safety evidence and the optimized
controller's validation 10 CPU timings and fixed-rate rollouts. Figures 1--4
and table 1 cover tracking and safety; figures 6--7 and tables 3--5 cover the
best implemented controller's performance. Earlier timing comparisons are
omitted. All measured samples from the optimized benchmark are retained;
median, p95, and maximum are reported without selecting only the fastest
samples. No simulations or timings were collected by these plotting scripts.

## Regenerate the complete figure set

```bash
conda activate xarm
cd /home/sriramk/xarm_ws
python scripts/figures/generate_figures.py
```

This runs the validation 9 plotter followed by the validation 10 plotter.
PDFs are vector exports with embedded fonts; PNGs are 600 dpi. Figures 6 and 7
target IEEE two-column width (7.16 inches) with 7--8 pt serif text. Use
`figure*` and `width=\\textwidth` for these figures. CSV and booktabs LaTeX table
fragments are provided; tables 3--5 are intended for two-column table placement.
The original validation 9 regeneration instructions below rebuild only that
portion and replace the README; use the combined command above for the full set.

## Updated performance: validation 10

![Optimized controller timing](fig06_optimized_timing.png)

[Vector PDF](fig06_optimized_timing.pdf). The optimized implementation uses a
seven-coordinate QP and fused JAX evaluation. It has {counts[0]} timed evaluations
({report['states_per_repeat']} saved states, {report['repeats']} repeats) after explicit warm-up.
Panel (a) uses a logarithmic time axis; panel (b) uses a linear axis.
The three timing scopes are nested, not additive: the density filter contains
the solver call, and nominal plus filter contains the filter. Bars are measured
statistics, not confidence intervals across independent experiments.

### Table 3: optimized CPU timing

{replay_table}

[CSV](table03_optimized_timing.csv) | [LaTeX](table03_optimized_timing.tex)

Median nominal-plus-filter time is **{report['reduced']['total_time']['median_ms']:.3f} ms**;
the solver alone has median **{report['reduced']['solver_solve_time']['median_ms']:.3f} ms**.

Nominal + filter timing includes reference generation, nominal control, JAX
evaluation and host synchronization, matrix construction, checks, QP update and
solve, and filter-return diagnostics. It excludes Bullet, rendering, sensor
input, and transport. Solver-only time is a narrower timed solver call. Do not
substitute solver-only time for the complete controller update.

## Fixed-rate results: validation 10

![Control and diagnostic loop time relative to the step budget](fig07_fixed_rate_budget.png)

[Vector PDF](fig07_fixed_rate_budget.pdf). The step budget is 20 ms at 50 Hz,
10 ms at 100 Hz (the new default), and 4.167 ms at 240 Hz. Timing statistics
are divided by each step budget; the dashed line marks 100% budget use.
Panel (a) includes nominal control and the filter. Panel (b) additionally
includes joint integration, Bullet stepping, mesh/contact queries, and the
logged clearance/tracking diagnostics. These are headless, unpaced kinematic
rollouts; GUI rendering and motor dynamics were not tested. No trajectory
samples are discarded after warm-up.

### Table 4: fixed-rate timing

{rate_table}

Control p95, control max, and loop max are in milliseconds. Misses count
individual measured updates exceeding that rate's simulation step.
[CSV](table04_fixed_rate_timing.csv) | [LaTeX](table04_fixed_rate_timing.tex)

### Table 5: fixed-rate clearance and convergence

{safety_table}

Sphere and mesh columns are minimum obstacle clearances. Final error is TCP
distance to the goal. Contacts count steps with obstacle contact. Floor count
is density-floor activation. All QPs returned successfully in these rollouts;
infeasible solves would abort the benchmark. This is not self-collision or
hardware validation.
[CSV](table05_fixed_rate_safety.csv) | [LaTeX](table05_fixed_rate_safety.tex)

The three rates reached the target without obstacle contacts, density-floor
activation, or measured deadline misses. This establishes the reported finite
run outcomes, not hard real-time bounds or robustness across different scenes.
The 100 Hz setting leaves more compute headroom for future sensing and GUI work.

## Provenance and evidence status

Processor: **{report['environment']['cpu']}**. JAX device(s):
`{', '.join(report['environment']['jax_devices'])}`; requested BLAS threads:
{report['environment']['blas_threads_requested']}. This is a CPU study; GPU comparison is pending.
Source: `results/cdf_qp_validations/validation_10/performance/`. Input hashes,
statistics, configuration context, and raw-run references are recorded in
[validation10_summary.json](validation10_summary.json) and the benchmark's
[full report](../../results/cdf_qp_validations/validation_10/performance/README.md).
The plotter recomputes timing summaries from the raw NPZ arrays and checks them
against benchmark.json before exporting figures.

The four original evidence requests remain listed below. Their timing request
is now partly addressed by this optimized CPU benchmark, logged
processor details and density-floor counts, and fresh fixed-rate rollouts.
Multiple scene/start trials, GPU measurements, and a matched OSCBF comparison
remain pending. Validation 9 tracking evidence remains unchanged.

{marker}

"""
    # Retain tracking evidence while omitting superseded timing presentation.
    start = previous.find("## Recorded QP timing")
    if start >= 0:
        end = previous.find("\n## ", start + 4)
        previous = previous[:start] + (previous[end:] if end >= 0 else "")
    previous = previous.replace("# Validation 9: paper figures and evidence", "## Validation 9 tracking and safety", 1)
    previous = previous.replace("## Original validation 9 archive", "## Validation 9 tracking and safety", 1)
    previous = previous.replace("## Requested evidence and remaining work", "## Original evidence checklist (validation 9 scope)", 1)
    checklist_start = previous.find("4. **Repeatability and timing.**")
    checklist_end = previous.find("\nValidation 10's", checklist_start)
    if checklist_start >= 0 and checklist_end >= 0:
        previous = previous[:checklist_start] + (
            "4. **Repeatability and timing.** Optimized timings, processor details,\n"
            "   density-floor counts, and fixed-rate checks are reported above.\n"
            "   Multiple starts/scenes, GPU comparison, and repeated independent\n"
            "   trials remain pending.\n"
        ) + previous[checklist_end:]
    for name in ("fig05_recorded_timing.pdf", "fig05_recorded_timing.png",
                 "table02_timing.csv", "table02_timing.tex"):
        (folder / name).unlink(missing_ok=True)
    readme_path.write_text(prefix + previous)
    print(f"Updated figures 6--7, tables 3--5, provenance, and README in {folder}")


if __name__ == "__main__":
    main()
