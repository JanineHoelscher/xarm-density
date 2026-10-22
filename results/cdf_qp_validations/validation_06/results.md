# Validation 6 results

## Goal

Use the validation-5 scene while enforcing density safety for every link sphere.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `697` |
| `simulated_time_s` | `2.904166666666667` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.0349654474428323` |
| `minimum_modeled_clearance_m` | `0.0341805996739102` |
| `minimum_link_sphere_clearance_m` | `0.0341805996739102` |
| `minimum_pybullet_mesh_clearance_m` | `0.050432385547675526` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.010000000001397563` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6999999999458825` |
| `maximum_jax_bullet_fk_error_m` | `6.176615371382485e-08` |
| `mean_qp_time_ms` | `11.400786347189849` |
| `maximum_qp_time_ms` | `29.71160301240161` |
| `mean_solver_time_ms` | `7.524504930616412` |

## Interpretation

The robot reached the target with all link spheres included. The minimum all-sphere clearance was `0.0341805996739102` m and Bullet reported contact on `0` steps.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
