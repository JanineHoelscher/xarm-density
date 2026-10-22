# Validation 7: Cluttered multi-sphere whole-arm CDF-QP

Find a collision-free whole-arm path through a deterministic cluttered workspace containing several spherical obstacles.

![PyBullet playback](pybullet.gif)

![Validation diagnostics](result.png)

## Setup

| Parameter | Value |
|---|---:|
| Timestep | `0.00416667 s` |
| Filter | `cdf` |
| Safety geometry | `whole_body` |
| Joint-rate limit | `0.7 rad/s` |
| Obstacles | `4` |

The runner matches the OSCBF playground: the pose-task controller produces
`u_nom`, the selected filter returns `qdot`, and the state is advanced with
`q_next = q + dt*qdot` before `resetJointState` and one `stepSimulation` call.

## Finding

The whole-arm CDF-QP reached the target through a cluttered field of `4` spheres. The minimum all-sphere clearance was `0.03558539282120392` m and Bullet reported contact on `0` steps.

Detailed metrics are listed in [results.md](results.md).
