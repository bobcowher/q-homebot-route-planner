# tests/test_episode_buffer.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from episode_buffer import EpisodeBuffer
from buffer import ReplayBuffer
from goal_geometry import world_coords


def _make_replay():
    return ReplayBuffer(
        max_size=10000,
        input_shape=(3, 96, 96),
        input_device='cpu',
        output_device='cpu',
        goal_dim=4,
    )


def _dummy_compute_reward(ag, dg, info):
    return np.zeros(len(ag), dtype=np.float32)


def _pos(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def _store_walk(ep, obs, n):
    """n-step straight-line walk: step i moves (i*10, i*10) -> ((i+1)*10, (i+1)*10)."""
    for i in range(n):
        ep.store(obs, 0, 0.0, obs, False,
                 achieved_prev=_pos(i * 10, i * 10),
                 achieved_next=_pos((i + 1) * 10, (i + 1) * 10))


def test_send_to_transition_count():
    """10-step episode, future strategy.

    Original: 10 transitions.
    Hindsight per step i: min(K, steps remaining after i) — last step has no
    future, skipped. Expected count is derived from EpisodeBuffer.K so the
    test tracks K tuning.
    """
    n = 10
    expected = n + sum(min(EpisodeBuffer.K, n - 1 - i) for i in range(n))

    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, n)
    ep.send_to(rep, _pos(300, 400), _dummy_compute_reward)
    assert rep.mem_ctr == expected, f"expected {expected}, got {rep.mem_ctr}"


def test_send_to_clears_nothing_on_its_own():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, 1)
    ep.send_to(rep, _pos(100, 100), _dummy_compute_reward)
    assert len(ep) == 1, "send_to must not clear the buffer — caller does that"


def test_absolute_goal_vectors():
    """Stored goals must be absolute coords: [robot_x, robot_y, goal_x, goal_y] via world_coords."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)
    desired = _pos(300, 400)

    n = 3
    _store_walk(ep, obs, n)
    ep.send_to(rep, desired, _dummy_compute_reward)

    # Pass 1 originals are the first n stored transitions.
    for i in range(n):
        achieved_p = _pos(i * 10, i * 10)
        achieved_n = _pos((i + 1) * 10, (i + 1) * 10)
        expected_goal      = world_coords(achieved_p[0], achieved_p[1], desired[0], desired[1])
        expected_next_goal = world_coords(achieved_n[0], achieved_n[1], desired[0], desired[1])
        assert torch.equal(rep.goal_memory[i], torch.as_tensor(expected_goal)), \
            f"transition {i}: goal must be world_coords at achieved_prev"
        assert torch.equal(rep.next_goal_memory[i], torch.as_tensor(expected_next_goal)), \
            f"transition {i}: next_goal must be world_coords at achieved_next"


def test_hindsight_absolute_goal_vectors():
    """Hindsight goals are future achieved_next positions; stored vectors must
    be absolute coords relative to this transition's prev/next positions."""
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    _store_walk(ep, obs, 2)  # step 0: (0,0)->(10,10), step 1: (10,10)->(20,20)
    ep.send_to(rep, _pos(300, 400), _dummy_compute_reward)

    # Layout: 2 originals, then step 0's hindsight (step 1 has no future).
    hg = _pos(20, 20)  # step 1's achieved_next
    assert rep.mem_ctr == 3
    expected_goal      = world_coords(0, 0, hg[0], hg[1])
    expected_next_goal = world_coords(10, 10, hg[0], hg[1])
    assert torch.equal(rep.goal_memory[2], torch.as_tensor(expected_goal))
    assert torch.equal(rep.next_goal_memory[2], torch.as_tensor(expected_next_goal))


def test_hindsight_success_is_terminal():
    """Relabeled success (reward 1) must store done=True; reward 0 stores done=False.

    The env terminates on success, so hindsight transitions must match — otherwise
    targets bootstrap past the goal and inflate Q in hindsight data.
    """
    def _success_compute_reward(ag, dg, info):
        return np.ones(len(ag), dtype=np.float32)

    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    n = 3
    _store_walk(ep, obs, n)
    ep.send_to(rep, _pos(100, 100), _success_compute_reward)
    # Layout: 3 original (reward 0 via stored value, done False) then hindsight.
    # All hindsight rewards are 1.0 here -> all hindsight dones must be True.
    cnt = rep.mem_ctr
    assert cnt > n, "expected hindsight transitions beyond the originals"
    assert not rep.terminal_memory[:n].any(), "original transitions must keep done=False"
    assert rep.terminal_memory[n:cnt].all(), "hindsight successes must be terminal"

    # And reward-0 relabels stay non-terminal.
    ep2  = EpisodeBuffer()
    rep2 = _make_replay()
    _store_walk(ep2, obs, n)
    ep2.send_to(rep2, _pos(100, 100), _dummy_compute_reward)
    assert not rep2.terminal_memory[:rep2.mem_ctr].any(), "reward-0 relabels must stay done=False"


def test_send_to_original_reward_preserved():
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    ep.store(obs, 0, 7.0, obs, False,
             achieved_prev=_pos(0, 0), achieved_next=_pos(10, 10))
    ep.send_to(rep, _pos(100, 100), _dummy_compute_reward)

    # First stored transition is the original — reward must be 7.0
    assert float(rep.reward_memory[0]) == 7.0


def test_spin_penalty():
    from episode_buffer import SPIN_PENALTY, BLOCKED_PENALTY
    ep  = EpisodeBuffer()
    rep = _make_replay()
    obs = torch.zeros(3, 96, 96, dtype=torch.uint8)

    # Store 8 dummy straight transitions so history is considered full (index >= 8)
    motion_straight = np.zeros(12, dtype=np.float32)
    motion_straight[8 + 3] = 1.0  # net_disp = 1.0
    for i in range(8):
        ep.store(obs, 0, 0.0, obs, False,
                 achieved_prev=_pos(i*10, 0), achieved_next=_pos((i+1)*10, 0),
                 motion_prev=motion_straight)

    # 8: Straight line (no penalty)
    ep.store(obs, 0, 0.0, obs, False,
             achieved_prev=_pos(80, 0), achieved_next=_pos(90, 0),
             motion_prev=motion_straight)

    # 9: Blocked pin (BLOCKED_PENALTY, no SPIN_PENALTY)
    motion_blocked = np.zeros(12, dtype=np.float32)
    ep.store(obs, 0, 0.0, obs, False,
             achieved_prev=_pos(90, 0), achieved_next=_pos(90, 0),
             motion_prev=motion_blocked)

    # 10: Spin cycle (SPIN_PENALTY)
    motion_spin = np.zeros(12, dtype=np.float32)
    motion_spin[8 + 2] = 0.06
    motion_spin[8 + 3] = 0.08  # net_disp = 0.1 < 0.25
    ep.store(obs, 0, 0.0, obs, False,
             achieved_prev=_pos(90, 0), achieved_next=_pos(94, 0),
             motion_prev=motion_spin)

    # Clear the replay buffer, send
    ep.send_to(rep, _pos(1000, 1000), _dummy_compute_reward)

    # Assertions
    # 0 to 8: straight -> 0.0
    import pytest
    for idx in range(9):
        assert float(rep.reward_memory[idx]) == pytest.approx(0.0), f"idx {idx} should have no penalty"
    # 9: blocked -> BLOCKED_PENALTY
    assert float(rep.reward_memory[9]) == pytest.approx(BLOCKED_PENALTY)
    # 10: spin -> SPIN_PENALTY
    assert float(rep.reward_memory[10]) == pytest.approx(SPIN_PENALTY)
