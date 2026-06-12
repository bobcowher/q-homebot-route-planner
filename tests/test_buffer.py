# tests/test_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from buffer import ReplayBuffer


def _make_buf():
    return ReplayBuffer(
        max_size=100,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
    )


def _dummy_transition(buf, goal, next_goal):
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    buf.store_transition(obs, 3, 1.0, obs, False, goal, next_goal)


def test_goals_stored_and_sampled():
    buf       = _make_buf()
    goal      = np.array([100.0, 200.0], dtype=np.float32)
    next_goal = np.array([90.0, 190.0], dtype=np.float32)
    for _ in range(20):
        _dummy_transition(buf, goal, next_goal)
    _, _, _, _, _, goals, next_goals = buf.sample_buffer(10)
    assert goals.shape == (10, 2), f"expected (10,2), got {goals.shape}"
    assert next_goals.shape == (10, 2)
    assert torch.allclose(goals, torch.tensor([100.0, 200.0]).expand(10, 2))
    assert torch.allclose(next_goals, torch.tensor([90.0, 190.0]).expand(10, 2))


def test_different_goals_round_trip():
    buf    = _make_buf()
    goal_a = np.array([10.0, 20.0], dtype=np.float32)
    goal_b = np.array([30.0, 40.0], dtype=np.float32)
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    for _ in range(10):
        buf.store_transition(obs, 0, 0.0, obs, False, goal_a, goal_b)
    for _ in range(10):
        buf.store_transition(obs, 0, 0.0, obs, False, goal_b, goal_a)
    _, _, _, _, _, goals, next_goals = buf.sample_buffer(20)
    assert goals.shape == (20, 2)
    assert next_goals.shape == (20, 2)
