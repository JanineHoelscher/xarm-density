# Validation 6: Whole-arm density

Use the validation-5 scene while enforcing density safety for every link sphere.

![PyBullet playback](pybullet.gif)

![Validation diagnostics](result.png)

## Setup

| Parameter | Value |
|---|---:|
| Timestep | `0.00416667 s` |
| Filter | `cdf` |
| Safety geometry | `whole_body` |
| Joint-rate limit | `0.7 rad/s` |
| Obstacles | `1` |

The runner matches the OSCBF playground: the pose-task controller produces
`u_nom`, the selected filter returns `qdot`, and the state is advanced with
`q_next = q + dt*qdot` before `resetJointState` and one `stepSimulation` call.

## Finding

The robot reached the target with all link spheres included. The minimum all-sphere clearance was `0.0341805996739102` m and Bullet reported contact on `0` steps.

Detailed metrics are listed in [results.md](results.md).
