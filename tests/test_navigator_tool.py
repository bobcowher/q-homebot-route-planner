# tests/test_navigator_tool.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unittest.mock import patch
import pytest
import math
import torch
import numpy as np
from planner.navigator_tool import NavigatorTool


class HeuristicMockModel(torch.nn.Module):
    """A mock PyTorch model that implements simple gradient-following navigation.

    This allows the NavigatorTool tests to run without loading a large checkpoint
    file from disk, and guarantees the robot will successfully reach target goals
    (like trash) to satisfy the test assertions.
    """
    def __init__(self):
        super().__init__()
        self.use_motion = False
        self.motion_window = 1
        self.macro_h = 1
        self.n_base = 8

    def forward(self, obs_t, goal_t, motion_t=None):
        # goal_t is [rx, ry, gx, gy]
        goal_val = goal_t[0].cpu().numpy()
        rx, ry, gx, gy = goal_val[0], goal_val[1], goal_val[2], goal_val[3]
        angle = math.atan2(gy - ry, gx - rx)
        # Map angle to HomeBot2D discrete actions:
        # Action 0: Up (dy=-4), Action 2: Right (dx=4), Action 4: Down (dy=4), Action 6: Left (dx=-4)
        act_idx = (int(round(angle / (math.pi / 4))) + 2) % 8
        q = torch.zeros(1, 8)
        q[0, act_idx] = 1.0
        return q


@pytest.fixture(autouse=True)
def mock_load_model():
    """Automatically patch load_q_model to return our HeuristicMockModel."""
    with patch("planner.navigator_tool.load_q_model") as mock_load:
        mock_load.return_value = HeuristicMockModel()
        yield mock_load


def test_reset_returns_state_and_go_to_returns_outcome_shape():
    nav = NavigatorTool()
    assert nav.env.render_mode == "rgb_array"  # default stays headless
    s = nav.reset(seed=0)
    assert "robot_xy" in s
    out = nav.go_to("human")
    assert {"reached", "steps", "state"} <= set(out)
    assert isinstance(out["reached"], bool) and isinstance(out["steps"], int)


def test_unknown_destination_is_error_not_crash():
    nav = NavigatorTool()
    nav.reset(seed=0)
    out = nav.go_to("bogus")
    assert out["reached"] is False and "error" in out


def test_state_delegates_to_world():
    nav = NavigatorTool()
    nav.reset(seed=0)
    assert nav.state() == nav.world.state()
    assert "trash_remaining" in nav.state()


def test_trash_reached_implies_actually_collected():
    # Honesty contract: reached=True for trash must mean the env collected it
    # (trash_remaining dropped), not merely "within the loose 79px fixture
    # threshold". The env only picks trash up within ~31px; the harness reach
    # must match, or it declares success 1.5 tiles short with trash untouched.
    nav = NavigatorTool()
    saw_reach = False
    for seed in range(4):
        before = nav.reset(seed=seed)["trash_remaining"]
        out = nav.go_to("trash")
        if out["reached"]:
            saw_reach = True
            assert out["state"]["trash_remaining"] < before, (
                f"seed {seed}: reached=True but trash_remaining stayed {before}")
    assert saw_reach, "navigator never reached trash in 4 seeds; cannot verify"
