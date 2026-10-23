# Recorded paper figures and reproducibility data

This folder contains seven paper figures, numerical plotters, tables, and raw
validation 9 tracking / validation 10 timing data. The ignored archive and old
ROS workspace are not required for replotting or compiling the paper.

## Replot from CSV

From the repository root:

```bash
conda activate xarm
python -m pip install -e '.[paper]'
python latex/figures/generate_figures.py --from-csv
```

To preserve the tracked exports while checking reproduction:

```bash
python latex/figures/generate_figures.py --output /tmp/xarm-paper-figures
```

The wrapper validates CSV/JSON checksums, reconstructs temporary NPZ inputs,
and runs both numerical plotters. It exports the numerical figures, tables,
and summaries. Figure 8 is a preserved/copied snapshot panel, not a new render.
PDF hashes can differ with timestamps, fonts, or plotting libraries; the data
is checksum-checked and numerical timing summaries are asserted by the plotters.

## Figures and tables

| Figure | Recorded evidence |
| --- | --- |
| [1: TCP tracking](fig01_tcp_tracking.pdf) | Reference and actual TCP trajectories |
| [2: Tracking errors](fig02_tracking_errors.pdf) | Position/orientation errors |
| [3: Joint velocities](fig03_joint_velocities.pdf) | Nominal and filtered commands |
| [4: Safety/intervention](fig04_safety_intervention.pdf) | Residual, clearances, contacts, intervention |
| [6: Optimized timing](fig06_optimized_timing.pdf) | Saved optimized timing samples and summary statistics |
| [7: Fixed-rate budgets](fig07_fixed_rate_budget.pdf) | Recorded 50/100/240 Hz timing budgets |
| [8: Robot snapshots](fig08_robot_snapshots.pdf) | Selected tight 1x4 recorded visual panel |

[Table 1](table01_performance.csv) summarizes tracking/safety.
[Table 3](table03_optimized_timing.csv) gives optimized timing statistics.
[Table 4](table04_fixed_rate_timing.csv) gives timing/deadlines.
[Table 5](table05_fixed_rate_safety.csv) gives clearance/contacts.
Matching booktabs `.tex` fragments and PNG previews are included.

See [raw data](data/README.md), [manifest](data/manifest.json),
[tracking summary](summary.json), and [timing summary](validation10_summary.json).
Metadata retains recorded settings and available processor/version information.
Timing scopes are nested, not additive; they are not worst-case guarantees.

## Scope

This reproduces plots/tables **from preserved measurements** and builds the
paper with [the LaTeX sources](../README.md). Historical baseline timing
arrays remain available for audit but are not plotted as headline results.
The wrapper does not smooth data or select only the fastest samples.

It does not rerun historical controllers or recreate the snapshot camera
session. New standalone simulations have a different CSV schema. Historical
runners remain in the local archive/Git history, not as active dependencies
of this build. These artifacts do not establish a matched OSCBF comparison
or multi-scene repeatability.
