# Validation 2: Obstacle-free CDF-QP

Check that the CDF-QP tracks the OSCBF nominal command without obstacles.

![PyBullet playback](pybullet.gif)

![Validation diagnostics](result.png)

## Setup

| Parameter | Value |
|---|---:|
| Timestep | `0.00416667 s` |
| Filter | `cdf` |
| Safety geometry | `none` |
| Joint-rate limit | `0.7 rad/s` |
| Obstacles | `0` |

The runner matches the OSCBF playground: the pose-task controller produces
`u_nom`, the selected filter returns `qdot`, and the state is advanced with
`q_next = q + dt*qdot` before `resetJointState` and one `stepSimulation` call.

## Finding

The robot reached the target. Bullet reported contact on `0` steps, and all configured QPs were feasible: `True`.

Detailed metrics are listed in [results.md](results.md).
