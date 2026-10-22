# Validation 2 results

## Goal

Check that the CDF-QP tracks the OSCBF nominal command without obstacles.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `455` |
| `simulated_time_s` | `1.8958333333333333` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.03497245425247386` |
| `minimum_modeled_clearance_m` | `None` |
| `minimum_link_sphere_clearance_m` | `None` |
| `minimum_pybullet_mesh_clearance_m` | `None` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.010000028525211044` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6860693570075146` |
| `maximum_jax_bullet_fk_error_m` | `5.272051141713389e-08` |
| `mean_qp_time_ms` | `11.501880982381568` |
| `maximum_qp_time_ms` | `13.30216697533615` |
| `mean_solver_time_ms` | `7.571523512040705` |

## Interpretation

The robot reached the target. Bullet reported contact on `0` steps, and all configured QPs were feasible: `True`.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
