# Validation 1 results

## Goal

Verify the unchanged OSCBF nominal controller and PyBullet playback.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `446` |
| `simulated_time_s` | `1.8583333333333334` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.03490595999512912` |
| `minimum_modeled_clearance_m` | `None` |
| `minimum_link_sphere_clearance_m` | `None` |
| `minimum_pybullet_mesh_clearance_m` | `None` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `None` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.7` |
| `maximum_jax_bullet_fk_error_m` | `6.115483367515731e-08` |
| `mean_qp_time_ms` | `0.0` |
| `maximum_qp_time_ms` | `0.0` |
| `mean_solver_time_ms` | `0.0` |

## Interpretation

The robot reached the target. Bullet reported contact on `0` steps, and all configured QPs were feasible: `True`.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
