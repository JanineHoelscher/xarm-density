"""Compact diagnostic plots and result serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sphere(ax, center: np.ndarray, radius: float, color: str = "0.45") -> None:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 14)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.32, linewidth=0)


def _equal_3d(ax, points: np.ndarray, padding: float = 0.12) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = 0.5 * (low + high)
    radius = max(0.2, 0.5 * np.max(high - low) + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 0.8))


def _diagnostics(
    axes: tuple[Any, Any, Any],
    time: np.ndarray,
    error: np.ndarray,
    clearance: np.ndarray,
    residual: np.ndarray,
) -> None:
    axes[0].plot(time, error, color="#0072B2")
    axes[0].set_ylabel("Target error [m]")
    finite_clearance = np.isfinite(clearance)
    if np.any(finite_clearance):
        axes[1].plot(
            time[1:][finite_clearance],
            clearance[finite_clearance],
            color="#009E73",
        )
    else:
        axes[1].text(
            0.5, 0.5, "No obstacles", ha="center", va="center",
            transform=axes[1].transAxes,
        )
    axes[1].axhline(0.0, color="0.3", ls="--", lw=0.9)
    axes[1].set_ylabel("Clearance [m]")
    axes[2].plot(time[1:], residual, color="#D55E00")
    axes[2].axhline(0.0, color="0.3", ls="--", lw=0.9)
    axes[2].set_ylabel("CDF residual")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(color="0.88", lw=0.7)


def save_point_result(
    result: dict[str, Any], config: Any, output_dir: Path, title: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    states = result["states"]
    goal = np.asarray(config.goal)
    time = np.arange(states.shape[0]) * config.dt
    fig = plt.figure(figsize=(8.0, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.45, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    diagnostics = tuple(fig.add_subplot(grid[i, 1]) for i in range(3))
    ax3d.plot(*states.T, color="#0072B2", lw=2.0)
    ax3d.scatter(*states[0], color="black", s=28, label="start")
    ax3d.scatter(*goal, marker="*", color="#009E73", s=90, label="target")
    for center, radius in zip(config.obstacle_centers, config.obstacle_radii):
        _sphere(ax3d, np.asarray(center), radius)
    scene = np.vstack([states, goal[None, :], np.asarray(config.obstacle_centers).reshape(-1, 3)])
    _equal_3d(ax3d, scene)
    ax3d.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=title)
    ax3d.zaxis.labelpad = -5
    ax3d.legend(frameon=True, loc="upper left")
    error = np.linalg.norm(states - goal, axis=1)
    _diagnostics(
        diagnostics, time, error, result["clearances"], result["residuals"]
    )
    fig.savefig(output_dir / "result.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_data(result, config, output_dir)


def save_robot_result(
    result: dict[str, Any], config: Any, output_dir: Path, title: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    states = result["ee_states"]
    goal = np.asarray(config.goal)
    time = np.arange(states.shape[0]) * config.dt
    fig = plt.figure(figsize=(8.0, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.45, 1.0))
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    diagnostics = tuple(fig.add_subplot(grid[i, 1]) for i in range(3))
    ax3d.plot(*states.T, color="#0072B2", lw=2.0, label="end effector")
    ax3d.scatter(*states[0], color="black", s=28, label="start")
    ax3d.scatter(*goal, marker="*", color="#009E73", s=90, label="target")
    for center, radius in zip(config.obstacle_centers, config.obstacle_radii):
        _sphere(ax3d, np.asarray(center), radius)
    for collision, color, alpha in (
        (result["start_collision_data"], "#CC79A7", 0.18),
        (result["final_collision_data"], "#56B4E9", 0.26),
    ):
        ax3d.scatter(
            collision[:, 0], collision[:, 1], collision[:, 2],
            color=color, s=16, alpha=alpha,
        )
    scene = np.vstack(
        [
            states,
            goal[None, :],
            result["start_collision_data"][:, :3],
            result["final_collision_data"][:, :3],
            np.asarray(config.obstacle_centers).reshape(-1, 3),
        ]
    )
    _equal_3d(ax3d, scene)
    ax3d.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=title)
    ax3d.legend(frameon=True, loc="upper left")
    error = np.linalg.norm(states - goal, axis=1)
    _diagnostics(
        diagnostics,
        time,
        error,
        result["whole_body_clearances"],
        result["residuals"],
    )
    fig.savefig(output_dir / "result.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_data(result, config, output_dir)


def _save_data(result: dict[str, Any], config: Any, output_dir: Path) -> None:
    payload = {"config": _jsonable(asdict(config)), "metrics": result["metrics"]}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(output_dir / "trajectory.npz", **arrays)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
