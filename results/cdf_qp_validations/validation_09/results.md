# Validation 9 results

## Goal

Track a smooth Cartesian reference with the closed-gripper fingertip midpoint while the whole-arm CDF-QP avoids a dense field of larger spherical obstacles.

## Metrics

| Metric | Value |
|---|---:|
| `steps` | `1680` |
| `simulated_time_s` | `7.0` |
| `target_reached` | `True` |
| `final_target_error_m` | `0.004838891979642508` |
| `maximum_reference_tracking_error_m` | `0.11537349751658607` |
| `final_orientation_error_deg` | `1.587790894284352` |
| `maximum_orientation_error_deg` | `16.055062202537407` |
| `minimum_modeled_clearance_m` | `0.0203695756485332` |
| `minimum_link_sphere_clearance_m` | `0.020369575648533228` |
| `minimum_pybullet_mesh_clearance_m` | `0.04854463049374312` |
| `pybullet_contact_steps` | `0` |
| `minimum_cdf_residual` | `0.00999999988134002` |
| `all_qps_feasible` | `True` |
| `maximum_joint_speed_rad_s` | `0.6840598713390065` |
| `maximum_jax_bullet_fk_error_m` | `3.275708725182709e-08` |
| `mean_qp_time_ms` | `10.626443949877284` |
| `maximum_qp_time_ms` | `14.892632141709328` |
| `mean_solver_time_ms` | `7.575295109667683` |

## Interpretation

The closed-gripper TCP CDF-QP reached the final pose after tracking a minimum-jerk Cartesian reference through a `5`-sphere clutter field. The minimum arm-and-gripper sphere clearance was `0.020369575648533228` m, the final orientation error was `1.587790894284352` deg, and Bullet reported contact on `0` steps.

The Bullet clearance is measured against the visual/collision URDF meshes,
while the modeled clearance uses the conservative JAX link-sphere model. They
are related safety checks, not geometrically identical quantities.
