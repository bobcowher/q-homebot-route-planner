"""Geometry helpers for the honest greedy reacher eval.

Stateless and unit-testable without a gym import.
"""
import math
import numpy as np

# Env trash pickup: robot.RADIUS(15) + tile_size(32) * _TRASH_RANGE(0.5) = 31 px.
# Match it so eval "reached" agrees with the env's own pickup distance.
GOAL_RADIUS = 31.0
ROBOT_STEP_PX = 4.0      # homebot DISCRETE_SPEED
EVAL_BUDGET_MULT = 3


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points in pixel space."""
    return math.hypot(bx - ax, by - ay)


def ego_vector(rx: float, ry: float, rtheta: float, gx: float, gy: float) -> np.ndarray:
    """Goal displacement expressed in the robot's egocentric frame.

    Rung 2 variable. Takes the world displacement (gx-rx, gy-ry) and rotates it
    by -rtheta into the robot frame: x = forward component, y = left component.
    MAGNITUDE IS PRESERVED (this is a rotation) — range still leaks through, so
    this isolates the allocentric->egocentric change from the range-stripping
    that Rung 3 adds on top.
    """
    dx = gx - rx
    dy = gy - ry
    c, s = math.cos(rtheta), math.sin(rtheta)
    x_ego = dx * c + dy * s
    y_ego = -dx * s + dy * c
    return np.array([x_ego, y_ego], dtype=np.float32)


def eval_step_budget(init_dist: float) -> int:
    """Step budget for greedy eval.

    Budget = EVAL_BUDGET_MULT * ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX).
    Tight enough that a circling policy cannot sweep-and-pass; grows linearly
    with spawn distance so far-away goals aren't penalised unfairly.
    """
    return EVAL_BUDGET_MULT * max(1, math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX))
