# CDF-QP sphere-world validations

These scripts build the density-QP in
`/home/sriramk/density-drift` into progressively more realistic tests for the
UF850. Run them from the `xarm` conda environment.

```bash
conda activate xarm
python scripts/validation_01_point_reach.py
python scripts/validation_02_point_one_sphere.py
python scripts/validation_03_point_multiple_spheres.py
python scripts/validation_04_uf850_reach.py
python scripts/validation_05_ee_only_safety.py
python scripts/validation_06_whole_arm_safety.py
```

`scripts/run_cdf_validations.py` runs the complete sequence. Each validation
writes `result.png`, `metrics.json`, and `trajectory.npz` under
`results/cdf_qp_validations/validation_XX`.

## Controller

The safety controller uses the pointwise CDF condition with a quadratic
weighted-flux parameterization. For state \(z\), let

\[
w_j(z)=\psi(z)\Phi_2(z)^\top\theta_j,\qquad
u_j(z)=\frac{w_j(z)}{\rho(z)},\qquad
\rho(z)=\frac{\psi(z)}
{\left(\lVert x_{\mathrm{ee}}(z)-x_g\rVert_2^2+\epsilon\right)^\alpha}.
\]

Here \(\Phi_2\) contains all total-degree monomials through degree two and
\(\psi\) is a product of compactly supported \(C^\infty\) obstacle bumps.
JAX differentiates \(w_j\) with respect to the current state, including the
UF850 forward kinematics and link-sphere geometry. The cached Clarabel QP
chooses \(\theta\) to track a bounded nominal Cartesian target command while
enforcing the pointwise CDF residual and joint/input bounds.

The nominal command is the target-seeking counterpart of `ustab`:

\[
v_{\mathrm{nom}} =
\operatorname{sat}_{v_{\max}}\!\left(K_p(x_g-x_{\mathrm{ee}})\right).
\]

It supplies the goal-directed behavior. The CDF constraint reshapes that
command when safety requires it; CDF feasibility alone does not select a useful
motion for this driftless robot.

## Validation intent

| ID | Model | Safety representation | Check |
|---:|---|---|---|
| 1 | 3-D point | none | Nominal reach and QP plumbing |
| 2 | 3-D point | one sphere | Single-obstacle avoidance |
| 3 | 3-D point | three spheres | Multiple simultaneous bumps |
| 4 | UF850 | none | JAX kinematics and Cartesian reach |
| 5 | UF850 | end effector only | Exposes unmodeled link collision |
| 6 | UF850 | all link spheres | Maintains whole-arm clearance |

Validations 5 and 6 deliberately use the same start, target, and obstacle.
The obstacle lies in the swept path of an intermediate link but remains away
from the end-effector path, making the modeling distinction observable.

## Baseline results

| ID | Reached | Minimum clearance | Minimum CDF residual | QPs feasible |
|---:|:---:|---:|---:|:---:|
| 1 | yes | no obstacles | 0.0200 | yes |
| 2 | yes | 0.0181 m | 0.0200 | yes |
| 3 | yes | 0.0177 m | 0.0200 | yes |
| 4 | yes | no obstacles | 0.0100 | yes |
| 5 | yes | **-0.0260 m whole arm** | 0.0100 | yes |
| 6 | yes | **0.0423 m whole arm** | 0.0100 | yes |

Validation 5 also maintains 0.105 m clearance for the end-effector geometry
that its density actually models. Its negative whole-arm value is therefore
the intended counterexample, not a QP failure. Validation 6 changes only the
safety model and removes that collision.

## Plots

The plots report target error, physical clearance, and the pointwise CDF
residual.

### Validation 1: point reach without obstacles

This isolates target convergence and basic QP feasibility before activating
the density safety window.

![Validation 1 plot](results/cdf_qp_validations/validation_01/result.png)

### Validation 2: point reach with one sphere

The direct target path intersects one obstacle. The density window bends the
trajectory while keeping positive clearance and residual.

![Validation 2 plot](results/cdf_qp_validations/validation_02/result.png)

### Validation 3: point reach with multiple spheres

Three overlapping sensing regions test the product bump and simultaneous
spatial derivatives.

![Validation 3 plot](results/cdf_qp_validations/validation_03/result.png)

### Validation 4: UF850 reach without obstacles

This activates JAX forward kinematics, the Cartesian nominal objective, and
the six-channel quadratic flux parameterization without safety geometry.

![Validation 4 plot](results/cdf_qp_validations/validation_04/result.png)

### Validation 5: end-effector-only safety

The modeled end effector stays clear, but an unmodeled intermediate link
intersects the obstacle. This is the deliberate failure case that motivates
whole-arm geometry.

![Validation 5 plot](results/cdf_qp_validations/validation_05/result.png)

### Validation 6: whole-arm safety

All link spheres enter the density window. The controller modifies the motion
enough to preserve positive whole-arm clearance while reaching the same target.

![Validation 6 plot](results/cdf_qp_validations/validation_06/result.png)

## TODO: PyBullet validation

- Integrate the CDF-QP with the ROS-free UF850 PyBullet loop used by
  `scripts/uf850_oscbf_pybullet_playground.py`.
- Use the same UF850 URDF, initial joint configuration, Cartesian target,
  obstacle geometry, control period, joint-rate limits, and link-sphere model
  for CDF-QP and OSCBF.
- Apply each computed joint-rate command through the same PyBullet stepping
  interface rather than only integrating \(q_{k+1}=q_k+\Delta t\,\dot q_k\).
- Reproduce validations 4--6 in PyBullet: obstacle-free reach,
  end-effector-only safety, and whole-arm safety.
- Check agreement between JAX kinematic clearance and PyBullet closest-point
  or contact queries at every control step.
- Verify target convergence, minimum physical clearance, CDF residual, QP
  feasibility, input limits, and mean/worst-case controller time.
- Run the matching OSCBF cases and report both controllers with identical
  metrics and scene settings.
- Save PyBullet recordings only after the numerical and physical collision
  checks agree.
