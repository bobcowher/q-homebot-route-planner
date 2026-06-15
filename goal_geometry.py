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


def world_vector(rx: float, ry: float, gx: float, gy: float) -> np.ndarray:
    """Goal displacement in the WORLD frame (no heading rotation).

    The discrete action space is 8 fixed compass directions (world-frame), and the
    observation viewport is north-up (world-frame). ego_vector rotated the goal by
    -heading into a frame matching neither — and heading isn't even an input — so
    the net got a goal direction scrambled by an unobservable rotation. Returning
    the raw world displacement puts goal, image, and actions all in one frame.
    """
    return np.array([gx - rx, gy - ry], dtype=np.float32)


def world_coords(rx: float, ry: float, gx: float, gy: float) -> np.ndarray:
    """Goal as raw ABSOLUTE coordinates: [robot_x, robot_y, goal_x, goal_y].

    Coordinate reframing (vs world_vector). Instead of handing the network the
    pre-computed displacement (gx-rx, gy-ry), feed both the robot's absolute pose
    and the goal's absolute position and let the net learn the relationship. The
    egocentric viewport hides absolute robot position from the image, so the pose
    must come in through this vector or the net can't localize itself. A real
    robot knows its own pose (odometry/localization) and the goal coordinate, so
    this stays within the env-realism rule. More compositional than world_vector
    (the net computes the subtraction + direction), which is why it's the natural
    place to test head depth.
    """
    return np.array([rx, ry, gx, gy], dtype=np.float32)


def eval_step_budget(init_dist: float) -> int:
    """Step budget for greedy eval.

    Budget = EVAL_BUDGET_MULT * ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX).
    Tight enough that a circling policy cannot sweep-and-pass; grows linearly
    with spawn distance so far-away goals aren't penalised unfairly.
    """
    return EVAL_BUDGET_MULT * max(1, math.ceil(max(init_dist, GOAL_RADIUS) / ROBOT_STEP_PX))
