# Validation 7 results

## Goal

Find a collision-free whole-arm path through a deterministic cluttered workspace containing several spherical obstacles.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `3570` |
| `simulated_time_s` | `14.875` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.034981542424898746` |
| `minimum_modeled_clearance_m` | `0.03558539282120392` |
| `minimum_link_sphere_clearance_m` | `0.03558539282120392` |
| `minimum_pybullet_mesh_clearance_m` | `0.036329645690564` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.009999999915383246` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.699999999945824` |
| `maximum_jax_bullet_fk_error_m` | `6.726863441970837e-08` |
| `mean_qp_time_ms` | `9.99783890860557` |
| `maximum_qp_time_ms` | `18.974958977196366` |
| `mean_solver_time_ms` | `6.909017892311788` |

## Interpretation

The whole-arm CDF-QP reached the target through a cluttered field of `4` spheres. The minimum all-sphere clearance was `0.03558539282120392` m and Bullet reported contact on `0` steps.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
