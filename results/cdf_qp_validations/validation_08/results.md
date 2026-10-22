# Validation 8 results

## Goal

Validate a collision-free path through a cluttered workspace with a fixed-open xArm gripper and gripper safety spheres.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `2719` |
| `simulated_time_s` | `11.329166666666666` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.03498818828755658` |
| `minimum_modeled_clearance_m` | `0.039680230299457775` |
| `minimum_link_sphere_clearance_m` | `0.039680230299457775` |
| `minimum_pybullet_mesh_clearance_m` | `0.0642288473432216` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.009999999974419907` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6999999989456487` |
| `maximum_jax_bullet_fk_error_m` | `6.310319584161325e-08` |
| `mean_qp_time_ms` | `9.194365414271372` |
| `maximum_qp_time_ms` | `14.351027028169483` |
| `mean_solver_time_ms` | `5.958975164188739` |

## Interpretation

The gripper-equipped whole-arm CDF-QP reached the target through a `3`-sphere clutter field sized for the larger gripper footprint. The minimum arm-and-gripper sphere clearance was `0.039680230299457775` m and Bullet reported contact on `0` steps.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
