# Validation 4 results

## Goal

Test the product density window with deterministic spherical obstacles.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `613` |
| `simulated_time_s` | `2.5541666666666667` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.03493256920081944` |
| `minimum_modeled_clearance_m` | `0.034831362451792074` |
| `minimum_link_sphere_clearance_m` | `0.034831362451792074` |
| `minimum_pybullet_mesh_clearance_m` | `0.011151410629092022` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.010000000000698273` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6999999999221674` |
| `maximum_jax_bullet_fk_error_m` | `5.829852034273475e-08` |
| `mean_qp_time_ms` | `13.71957762960685` |
| `maximum_qp_time_ms` | `28.27091299695894` |
| `mean_solver_time_ms` | `9.70330783160294` |

## Interpretation

The robot reached the target. Bullet reported contact on `0` steps, and all configured QPs were feasible: `True`.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
