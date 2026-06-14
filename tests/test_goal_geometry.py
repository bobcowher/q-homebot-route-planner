# tests/test_goal_geometry.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import math
from goal_geometry import distance, eval_step_budget, GOAL_RADIUS, ROBOT_STEP_PX, EVAL_BUDGET_MULT


def test_distance_basic():
    assert abs(distance(0, 0, 3, 4) - 5.0) < 1e-9
    assert abs(distance(0, 0, 0, 0)) < 1e-9


def test_eval_step_budget_grows_with_distance():
    assert eval_step_budget(10.0) <= eval_step_budget(100.0) <= eval_step_budget(500.0)


def test_eval_step_budget_minimum():
    """Budget never drops below the GOAL_RADIUS floor scaled by mult."""
    assert eval_step_budget(0.0) >= EVAL_BUDGET_MULT


def test_eval_step_budget_formula():
    d = 100.0
    expected = EVAL_BUDGET_MULT * math.ceil(max(d, GOAL_RADIUS) / ROBOT_STEP_PX)
    assert eval_step_budget(d) == expected
