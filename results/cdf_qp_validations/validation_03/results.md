# Validation 3 results

## Goal

Avoid one sphere on the nominal workspace path using all link spheres.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `614` |
| `simulated_time_s` | `2.558333333333333` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.0349096768196977` |
| `minimum_modeled_clearance_m` | `0.02695310443353191` |
| `minimum_link_sphere_clearance_m` | `0.026953104433531938` |
| `minimum_pybullet_mesh_clearance_m` | `0.0026077739999621763` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.010000000000105716` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.699999999982385` |
| `maximum_jax_bullet_fk_error_m` | `5.6978000819376755e-08` |
| `mean_qp_time_ms` | `12.952907392343716` |
| `maximum_qp_time_ms` | `19.05141497263685` |
| `mean_solver_time_ms` | `9.64303386336229` |

## Interpretation

The robot reached the target. Bullet reported contact on `0` steps, and all configured QPs were feasible: `True`.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
