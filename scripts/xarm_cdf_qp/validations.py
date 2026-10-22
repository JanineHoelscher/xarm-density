"""Definitions and runners for the six staged CDF-QP validations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xarm_cdf_qp.plotting import save_point_result, save_robot_result
from xarm_cdf_qp.point import PointWorldConfig, simulate_point_world
from xarm_cdf_qp.robot import RobotWorldConfig, simulate_robot_world


DEFAULT_RESULTS = Path("/home/sriramk/xarm_ws/results/cdf_qp_validations")
Q_START = (0.0, -0.7, -1.0, 0.0, 0.8, 0.0)
ROBOT_GOAL = (0.40, 0.35, 0.45)


def validation_config(index: int) -> tuple[Any, str]:
    if index == 1:
        return (
            PointWorldConfig(
                start=(-0.55, -0.25, 0.25),
                goal=(0.55, 0.25, 0.55),
            ),
            "Validation 1: point reach",
        )
    if index == 2:
        return (
            PointWorldConfig(
                start=(-0.55, -0.25, 0.25),
                goal=(0.55, 0.25, 0.55),
                obstacle_centers=((0.0, 0.0, 0.40),),
                obstacle_radii=(0.14,),
            ),
            "Validation 2: one sphere",
        )
    if index == 3:
        return (
            PointWorldConfig(
                start=(-0.58, -0.30, 0.22),
                goal=(0.58, 0.30, 0.58),
                obstacle_centers=(
                    (-0.18, -0.10, 0.34),
                    (0.08, 0.08, 0.43),
                    (0.31, 0.19, 0.51),
                ),
                obstacle_radii=(0.11, 0.12, 0.10),
            ),
            "Validation 3: multiple spheres",
        )
    if index == 4:
        return (
            RobotWorldConfig(q_start=Q_START, goal=ROBOT_GOAL),
            "Validation 4: UF850 reach",
        )
    if index == 5:
        return (
            RobotWorldConfig(
                q_start=Q_START,
                goal=ROBOT_GOAL,
                obstacle_centers=((0.25, 0.15, 0.40),),
                obstacle_radii=(0.07,),
                safety_mode="ee",
                sensing_width=0.16,
            ),
            "Validation 5: end-effector-only density",
        )
    if index == 6:
        return (
            RobotWorldConfig(
                q_start=Q_START,
                goal=ROBOT_GOAL,
                obstacle_centers=((0.25, 0.15, 0.40),),
                obstacle_radii=(0.07,),
                safety_mode="whole_body",
                sensing_width=0.16,
            ),
            "Validation 6: whole-arm density",
        )
    raise ValueError("Validation index must be in 1..6")


def run_validation(index: int, output_root: Path = DEFAULT_RESULTS) -> dict[str, Any]:
    config, title = validation_config(index)
    output_dir = output_root / f"validation_{index:02d}"
    if isinstance(config, PointWorldConfig):
        result = simulate_point_world(config)
        save_point_result(result, config, output_dir, title)
    else:
        result = simulate_robot_world(config)
        save_robot_result(result, config, output_dir, title)
    return result
