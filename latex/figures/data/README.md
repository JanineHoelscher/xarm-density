# Preserved numerical inputs

These are raw recorded arrays exported with 17 significant digits.
`manifest.json` describes array names, column slices, shapes, dtypes, source
names, and CSV/JSON checksums. Restoration checks those hashes before creating
temporary NPZ plotting inputs. Original ZIP timestamps need not be reproduced.

| Prefix | Contents |
| --- | --- |
| `validation09_1681_samples` | Joint/TCP states, reference positions, orientation errors, including initial state |
| `validation09_1680_samples` | Commands, residuals, clearances, contacts, recorded timings |
| `validation10_replay_160_samples` | Saved-state indices |
| `validation10_replay_480_samples` | Full/reduced replay timing arrays over three repetitions |
| `validation10_100hz_700_samples`, `validation10_100hz_701_samples` | Recorded 100 Hz histories and states |
| `validation10_50hz_350_samples`, `validation10_50hz_351_samples` | Recorded 50 Hz histories and states |
| `validation10_240hz_1680_samples`, `validation10_240hz_1681_samples` | Recorded 240 Hz histories and states |

Metadata JSONs retain saved scenes, gains, timestep and available benchmark
environment details. The original validation 9 capture does not record its
processor/backend. These are historical measurements, not new package runs.

## Units and timestamps

- Joint positions: radians. Joint velocities: rad/s.
- TCP positions and clearances: meters. Orientation errors: radians.
- Controller/filter/solver/loop timings: seconds.
- Residuals: compared against the recorded margin, 0.01.
- State histories include their initial state at `time_s`.
- Step command/residual rows use `time_s`; post-integration tracking/clearance/
  contact diagnostics refer to `next_time_s`.
- Replay timestamps identify the saved source state, not elapsed benchmark time.
- Extra index/time columns are export annotations, not original arrays.

See [the figure README](../README.md) for regeneration commands and scope.
`export_reproducibility_data.py` retains a historical archive-export utility;
the supported paper workflow uses its `--restore-to` mode only.
