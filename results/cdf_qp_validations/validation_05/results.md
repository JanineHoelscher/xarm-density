# Validation 5 results

## Goal

Show that EE clearance does not imply clearance for intermediate links.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `455` |
| `simulated_time_s` | `1.8958333333333333` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.03497611688760336` |
| `minimum_modeled_clearance_m` | `0.08862639564883287` |
| `minimum_link_sphere_clearance_m` | `-0.035153319799761246` |
| `minimum_pybullet_mesh_clearance_m` | `-0.00616088061069546` |
| `pybullet_contact_steps` | `96` |
| `minimum_cdf_residual` | `0.01000000169009153` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6860693570075146` |
| `maximum_jax_bullet_fk_error_m` | `5.272051141713389e-08` |
| `mean_qp_time_ms` | `11.809386428289505` |
| `maximum_qp_time_ms` | `13.635832001455128` |
| `mean_solver_time_ms` | `7.8527483565267175` |

## Interpretation

The robot reached the target. The end-effector-only filter left the remaining links unmodeled; the minimum all-sphere clearance was `-0.035153319799761246` m and Bullet reported contact on `96` steps.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
