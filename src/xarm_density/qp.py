"""Reusable matrix-form control-density-function quadratic program."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import clarabel
import numpy as np
from scipy import sparse
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CDFQPConfig:
    """Numerical settings shared by pointwise and cloud CDF-QPs."""

    residual_margin: float = 0.0
    input_lower: float | FloatArray = -np.inf
    input_upper: float | FloatArray = np.inf


@dataclass(frozen=True)
class CDFQPData:
    """Finite-dimensional QP assembled at the current operating point.

    The CDF inequalities use ``cdf_matrix @ theta + drift >= margin`` and
    the applied input uses ``u = input_matrix @ theta``.
    """

    hessian: FloatArray
    linear_cost: FloatArray
    cdf_matrix: FloatArray
    drift: FloatArray
    input_matrix: FloatArray


@dataclass(frozen=True)
class CDFQPResult:
    theta: FloatArray
    control: FloatArray
    residuals: FloatArray
    status: str
    solve_time: float
    solver_solve_time: float
    solver_num_iters: int
    objective: float


class CDFQP:
    """Cached direct Clarabel realization of a fixed-shape online CDF-QP."""

    def __init__(self, n_theta: int, n_constraints: int, n_inputs: int, config: CDFQPConfig):
        self.config = config
        self.n_theta = n_theta
        self.n_constraints = n_constraints
        self.n_inputs = n_inputs
        self.lower = np.broadcast_to(config.input_lower, (n_inputs,)).astype(float)
        self.upper = np.broadcast_to(config.input_upper, (n_inputs,)).astype(float)
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("direct Clarabel CDF-QP requires finite input bounds")
        self._solver: clarabel.DefaultSolver | None = None

    @staticmethod
    def _upper_triangle(matrix: FloatArray) -> sparse.csc_matrix:
        """Return a fixed-pattern upper-triangular CSC matrix, including zeros."""

        rows, cols = np.triu_indices(matrix.shape[0])
        return sparse.csc_matrix((matrix[rows, cols], (rows, cols)), shape=matrix.shape)

    @staticmethod
    def _dense_csc(matrix: FloatArray) -> sparse.csc_matrix:
        """Return a fixed-pattern CSC matrix so Clarabel data updates remain valid."""

        rows, cols = np.indices(matrix.shape)
        return sparse.csc_matrix(
            (matrix.ravel(), (rows.ravel(), cols.ravel())),
            shape=matrix.shape,
        )

    def _problem_data(
        self,
        data: CDFQPData,
        hessian: FloatArray,
    ) -> tuple[sparse.csc_matrix, FloatArray, sparse.csc_matrix, FloatArray]:
        cdf_matrix = np.asarray(data.cdf_matrix, dtype=float)
        drift = np.asarray(data.drift, dtype=float)
        input_matrix = np.asarray(data.input_matrix, dtype=float)
        constraint_matrix = np.vstack((-cdf_matrix, input_matrix, -input_matrix))
        constraint_bound = np.r_[
            drift - self.config.residual_margin,
            self.upper,
            -self.lower,
        ]
        return (
            self._upper_triangle(hessian),
            np.asarray(data.linear_cost, dtype=float),
            self._dense_csc(constraint_matrix),
            constraint_bound,
        )

    def solve(self, data: CDFQPData) -> CDFQPResult:
        """Update the cached problem and solve one online control step."""

        hessian = 0.5 * (
            np.asarray(data.hessian, dtype=float)
            + np.asarray(data.hessian, dtype=float).T
        )
        eigenvalues = np.linalg.eigvalsh(hessian)
        if np.min(eigenvalues) < -1e-9:
            raise ValueError("CDF-QP Hessian must be positive semidefinite")

        start = perf_counter()
        p_matrix, q_vector, a_matrix, b_vector = self._problem_data(data, hessian)
        used_cached_solver = self._solver is not None
        if self._solver is None:
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            self._solver = clarabel.DefaultSolver(
                p_matrix,
                q_vector,
                a_matrix,
                b_vector,
                [clarabel.NonnegativeConeT(a_matrix.shape[0])],
                settings,
            )
        else:
            self._solver.update(P=p_matrix, q=q_vector, A=a_matrix, b=b_vector)
        solver_start = perf_counter()
        solution = self._solver.solve()
        solver_status = str(solution.status)
        if used_cached_solver and solver_status not in ("Solved", "AlmostSolved"):
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            self._solver = clarabel.DefaultSolver(
                p_matrix,
                q_vector,
                a_matrix,
                b_vector,
                [clarabel.NonnegativeConeT(a_matrix.shape[0])],
                settings,
            )
            solution = self._solver.solve()
        solver_elapsed = perf_counter() - solver_start
        elapsed = perf_counter() - start

        solver_status = str(solution.status)
        if solver_status not in ("Solved", "AlmostSolved"):
            raise RuntimeError(f"CDF-QP failed with status={solver_status}")

        theta = np.asarray(solution.x, dtype=float)
        input_matrix = np.asarray(data.input_matrix, dtype=float)
        cdf_matrix = np.asarray(data.cdf_matrix, dtype=float)
        drift = np.asarray(data.drift, dtype=float)
        return CDFQPResult(
            theta=theta,
            control=input_matrix @ theta,
            residuals=cdf_matrix @ theta + drift,
            status="optimal" if solver_status == "Solved" else "optimal_inaccurate",
            solve_time=elapsed,
            solver_solve_time=solver_elapsed,
            solver_num_iters=int(solution.iterations),
            objective=float(solution.obj_val),
        )
