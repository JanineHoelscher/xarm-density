# CDF-QP: physical and matrix formulations

This document describes the implemented driftless UF850 controller in
`scripts/xarm_cdf_qp/robot.py` and `core.py`. Validation 9 supplies a nominal
joint velocity from the operational-space pose controller and filters it.

## Version 1: density and weighted flux

Let $q\in\mathbb R^6$ and $\dot q=u$. For each robot collision sphere $a$ and
obstacle sphere $b$, define:

$$
d_{ab}(q)=\|p_a(q)-o_b\|-r_a-R_b.
$$

The density window and target-weighted density are

$$
\psi(q)=\prod_{a,b}S(d_{ab}(q)/\delta),\qquad
\rho(q)=\frac{\psi(q)}{(\|x_{\rm TCP}(q)-x_g\|^2+\epsilon)^\alpha}.
$$

Here $S$ is the smooth step used by the implementation: zero at and inside
collision, one beyond the sensing width. Gripper spheres are included when
enabled. One product combines all modeled pairs; there is not a separate CDF
inequality per sphere pair. Self-collision is not encoded by this product.

Set $z=q/\pi$ and collect all total-degree monomials through degree two:

$$
\Phi_2(q)=[1,z_1,\ldots,z_6,z_1^2,z_1z_2,\ldots,z_6^2]^T\in\mathbb R^{28},
\qquad b(q)=\psi(q)\Phi_2(q).
$$

For each channel $j$, choose coefficients $\theta_j\in\mathbb R^{28}$ and set

$$
w_j(q)=b(q)^T\theta_j,\qquad
u_j(q)=\frac{w_j(q)}{\rho(q)},\qquad
\theta=[\theta_1^T,\ldots,\theta_6^T]^T.
$$

The nominal command is evaluated outside this QP. In validation 9 it combines
Cartesian reference tracking, an orientation reference, and posture behavior.
The filter solves, at each current configuration:

$$
\begin{aligned}
\min_\theta\quad &
\frac12\|u_\theta-u_{\rm nom}\|^2
+\frac{\lambda_u}{2}\|u_\theta\|^2
+\frac{\lambda_\theta}{2}\|\theta\|^2\\
\text{subject to}\quad &
\sum_{j=1}^6\frac{\partial w_j}{\partial q_j}\ge\eta,\\
&-u_{\max}\mathbf1\le u_\theta\le u_{\max}\mathbf1.
\end{aligned}
$$

The derivative holds $\theta$ fixed while differentiating the parameterized
field. It is not the total derivative of the online optimizer $\theta^*(q)$.
There is no slack in this implementation and feasibility is not automatic.
There are joint velocity bounds in this QP, not joint position inequalities.

The density condition is imposed at the current configuration only. These
experiments report empirical clearance and residuals; pointwise feasibility
alone does not prove forward invariance or convergence for the recomputed,
sampled controller. The nominal controller is necessary for useful progress.

## Version 2: matrices passed to the QP solver

At the current configuration define

$$
\bar\rho=\max(\rho,10^{-10}),\qquad
B=I_6\otimes\frac{b^T}{\bar\rho}\in\mathbb R^{6\times168},
$$

$$
a^T=\left[
\left(\frac{\partial b}{\partial q_1}\right)^T,\ldots,
\left(\frac{\partial b}{\partial q_6}\right)^T
\right]\in\mathbb R^{1\times168}.
$$

Thus $u=B\theta$ and the implemented residual is $a^T\theta$.
The exact weighted-flux interpretation $w=\rho u$ holds where the density
floor is inactive. The floor prevents division by zero; it is not a collision
recovery mechanism.

Allow a general task map $T$ and desired task command $v$:

$$
\begin{aligned}
\min_\theta\quad &
\frac{\beta}{2}\|T\theta-v\|^2
+\frac{\lambda_u}{2}\|B\theta\|^2
+\frac{\lambda_\theta}{2}\|\theta\|^2\\
\text{subject to}\quad &a^T\theta\ge\eta,\qquad
-u_{\max}\mathbf1\le B\theta\le u_{\max}\mathbf1.
\end{aligned}
$$

For validation 9, $T=B$, $v=u_{\rm nom}$, $\beta=1$,
$\lambda_u=0.02$, $\lambda_\theta=0.0002$, $\eta=0.01$,
and $u_{\max}=0.7$ rad/s. The alternative controller branch, used when no
nominal joint command is passed, uses $T=J_x(q)B$, Cartesian nominal velocity
$v$, and $\beta=8$ instead.

Expanding the objective and dropping its constant term gives

$$
\min_\theta\ \tfrac12\theta^TH\theta+g^T\theta,
\quad H=\beta T^TT+\lambda_u B^TB+\lambda_\theta I,
\quad g=-\beta T^Tv.
$$

Clarabel receives the inequalities in upper-bound form:

$$
\begin{bmatrix}-a^T\\B\\-B\end{bmatrix}\theta
\le
\begin{bmatrix}-\eta\\u_{\max}\mathbf1\\u_{\max}\mathbf1\end{bmatrix}.
$$

There are 168 decision variables and 13 scalar inequalities: one density
condition and twelve joint-rate bounds. The positive coefficient penalty makes
the Hessian positive definite. JAX evaluates the basis and derivatives;
NumPy/SciPy assemble the matrices and CPU Clarabel solves a cached QP.

The simulator then applies $q_{k+1}=q_k+\Delta t\,B(q_k)\theta_k$ using Bullet
joint resets and records independent mesh clearance/contact queries.
