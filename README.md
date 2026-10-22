# xarm-density

Density-based safe motion control experiments for the UFactory UF850 arm.
The workspace contains a Stanford OSCBF baseline and a control density function
(CDF) quadratic-program controller, tested with PyBullet and differentiable
robot kinematics.

## Featured result: validation 9

A UF850 with its gripper closed tracks a smooth Cartesian reference through
five spherical obstacles. The controller tracks the `link_tcp` frame, includes
arm and gripper collision spheres in the density, and uses a fixed orientation
reference.

![UF850 whole-arm CDF-QP trajectory tracking](results/cdf_qp_validations/validation_09/pybullet.gif)

| Saved measurement | Validation 9 |
|---|---:|
| Final position error | 4.84 mm |
| Final orientation error | 1.59 degrees |
| Maximum reference tracking error | 115.4 mm |
| Minimum modeled sphere clearance | 20.4 mm |
| Minimum PyBullet mesh clearance | 48.5 mm |
| Obstacle contact steps | 0 |
| All QPs feasible | Yes |
| Mean measured QP pipeline time | 10.21 ms |
| Maximum measured QP pipeline time | 13.83 ms |
| Simulated trajectory duration | 7.0 s |

These are saved results, not a new benchmark. The runner integrates joint
velocity and sets Bullet joint positions directly. It does not validate motor
dynamics or hardware tracking. Its simulated timestep is 4.17 ms, shorter than
the reported mean QP pipeline time, so this result does not establish 240 Hz
wall-clock operation. Sphere clearance and mesh clearance measure different
geometries and need not agree exactly.

[Scenario details](results/cdf_qp_validations/validation_09/README.md) |
[All nine results](results/cdf_qp_validations/results.md) |
[Detailed CDF-QP mathematics](docs/CDF_QP.md)

## Run the featured example

On the existing development machine:

```bash
source /home/sriramk/anaconda3/etc/profile.d/conda.sh
conda activate xarm
cd /home/sriramk/xarm_ws
python scripts/pybullet_validation_09.py
```

The runner is headless. It saves `pybullet.gif`, `result.png`, `metrics.json`,
and `trajectory.npz` in `results/cdf_qp_validations/validation_09/`, replacing
existing outputs. Initial JAX compilation may take time.

Run all PyBullet validations with:

```bash
python scripts/run_pybullet_cdf_validations.py
```

## CDF-QP in physical notation

The state is the six joint angles $q$, and the command is joint velocity:

```math
\dot q=u,\qquad q\in\mathbb R^6.
```

For robot sphere $a$ and obstacle $b$, define surface clearance and a smooth
obstacle window:

```math
d_{ab}(q)=\|p_a(q)-o_b\|-r_a-R_b,\qquad
\psi(q)=\prod_{a,b}S\left(\frac{d_{ab}(q)}{\delta}\right).
```

$S$ is zero at nonpositive clearance and one beyond the sensing width
$\delta$. Thus every modeled link-obstacle pair contributes to the density:

```math
\rho(q)=\frac{\psi(q)}{(\|x_{\rm TCP}(q)-x_g\|^2+\epsilon)^\alpha}.
```

Parameterize the weighted flux $w=\rho u$ with quadratic basis functions,
then choose its coefficients to stay close to a nominal joint command:

```math
\begin{aligned}
\min_\theta\quad &
\frac12\|u_\theta(q)-u_{\rm nom}\|^2
+\frac{\lambda_u}{2}\|u_\theta(q)\|^2
+\frac{\lambda_\theta}{2}\|\theta\|^2\\
\text{subject to}\quad &
\nabla_q\cdot w_\theta(q)\ge\eta,\\
&-u_{\max}\le u_\theta(q)\le u_{\max}.
\end{aligned}
```

The nominal controller supplies progress along the reference and orientation
tracking. The QP adjusts its joint command through one aggregate density
condition. This is the pointwise condition used in these experiments; satisfying
it at sampled states alone is not a whole-workspace or sampled-data safety proof.

## CDF-QP in solver notation

At the current $q$, the applied command and transport residual are linear in
the coefficients: $u=B(q)\theta$ and $\nabla\cdot w=a(q)^T\theta$.
The same problem becomes:

```math
\begin{aligned}
\min_{\theta\in\mathbb R^{168}}\quad &
\frac12\theta^TH\theta+g^T\theta\\
\text{subject to}\quad &a^T\theta\ge\eta,\\
&-u_{\max}\mathbf1\le B\theta\le u_{\max}\mathbf1,
\end{aligned}
```

```math
H=B^TB+\lambda_u B^TB+\lambda_\theta I,\qquad
g=-B^Tu_{\rm nom}.
```

For validation 9, $\lambda_u=0.02$, $\lambda_\theta=0.0002$,
$\eta=0.01$, and $u_{\max}=0.7$ rad/s. There are 28 quadratic basis terms
per joint channel, hence 168 coefficients. JAX computes derivatives; Clarabel
solves the CPU QP. See the detailed mathematics for the definitions of $B$ and
$a$, numerical density flooring, and the Cartesian-objective variant.

## Layout and dependencies

- `scripts/uf850_oscbf_pybullet_playground.py`: original OSCBF playground.
- `scripts/xarm_cdf_qp/`: CDF-QP assembly, robot geometry, and validation runners.
- `scripts/pybullet_validation_01.py` through `09.py`: individual experiments.
- `envs/`: conda environment specification.
- `results/cdf_qp_validations/`: saved measurements, plots, and animations.
- `src/xarm_ros2/`: separately cloned vendor workspace; excluded from this repo.
- `resources/`: paper notes; downloaded PDFs are excluded from publication.

The current setup is machine-specific. The source expects the vendor URDF
workspace at `/home/sriramk/xarm_ws/src/xarm_ros2` and the CDF-QP backend at
`/home/sriramk/density-drift/src`. That backend is not included in this repository.
A fresh clone alone is therefore not yet sufficient to reproduce the CDF runs.
The environment specification also does not install that local backend.

Vendor source: https://github.com/xArm-Developer/xarm_ros2

Vendor commit used locally: `d0b95117dabd3883f41155125aa3f67d37901c18`.

For the existing machine, dependencies can be refreshed with:

```bash
conda env update -n xarm -f envs/xarm_oscbf_pybullet_environment.yml
```

See [scripts/README.md](scripts/README.md) for the original playground and URDF
generation commands. The older [validation notes](README_CDF_QP_VALIDATIONS.md)
describe the first six analytical tests; their PyBullet TODO predates the saved
nine-case suite.

## Next work

- [Validation 10 TODO](results/cdf_qp_validations/validation_10/TODO.md): GPU
  profiling, numerical optimization, interactive GUI, and CPU/GPU benchmarks.
- Package the local CDF backend, remove absolute paths, and document fresh setup.
- Compare CDF-QP and OSCBF with identical geometry, scenes, and timing settings.
- Check sphere coverage over gripper openings and add narrow-passage tests.
- Validate motor-driven tracking and measured control deadlines before hardware.
