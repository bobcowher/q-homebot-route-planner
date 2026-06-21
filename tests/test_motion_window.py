"""Windowed motion: the net-displacement term must make a *moving* limit cycle
(spinning) observable, while window<=1 reproduces the original velocity-only
vector bit-for-bit (so old checkpoints still load)."""
from motion import MotionState, make_motion, motion_dim
from goal_geometry import ROBOT_STEP_PX

A = 8       # action_dim used throughout
S = ROBOT_STEP_PX


def test_window_le_1_is_unchanged_width():
    # Default and explicit window=1 both give the original n+2 vector.
    assert motion_dim(A) == A + 2
    assert motion_dim(A, 1) == A + 2
    assert len(make_motion(A, 0, S, 0.0)) == A + 2
    assert len(MotionState(A).vec(0.0, 0.0)) == A + 2


def test_window_gt_1_adds_two_dims():
    assert motion_dim(A, 8) == A + 4
    assert len(make_motion(A, 0, S, 0.0, S, 0.0, window=8)) == A + 4
    assert len(MotionState(A, 8).vec(0.0, 0.0)) == A + 4


def test_make_motion_normalizes_net_by_window():
    # A full straight run over the window reads ~1 on the net dims.
    m = make_motion(A, 0, S, 0.0, 8 * S, 0.0, window=8)
    assert abs(m[A + 2] - 1.0) < 1e-6   # net_dx / (8*S)
    assert abs(m[A + 3]) < 1e-6


def test_spinning_collapses_the_windowed_net():
    # A 2-cycle: alternate between two poses one step apart. Per-step velocity is
    # full-size every step, but net displacement over the window is ~0.
    ms = MotionState(A, window=4)
    a, b = (0.0, 0.0), (S, 0.0)
    for pose, act in [(a, 0), (b, 1), (a, 0), (b, 1)]:
        ms.commit(pose[0], pose[1], act)
    m = ms.vec(a[0], a[1])              # continuing the cycle back to A
    velocity = (m[A], m[A + 1])
    net = (m[A + 2], m[A + 3])
    assert abs(velocity[0]) > 0.5       # still moving (looks healthy to 1-step)
    assert abs(net[0]) < 1e-6 and abs(net[1]) < 1e-6   # but going nowhere


def test_straight_run_keeps_a_large_windowed_net():
    # Monotonic progress: net accumulates across the window and reads ~1 once
    # normalized -- the opposite end of the signal from the spinning case (~0).
    # (Per-step velocity also reads ~1 here; the discriminator is net magnitude:
    # ~1 for a straight run vs ~0 for a cycle, both seen above.)
    ms = MotionState(A, window=4)
    for i in range(4):
        ms.commit(i * S, 0.0, 0)
    m = ms.vec(4 * S, 0.0)
    net_mag = abs(m[A + 2])
    assert net_mag > 0.9
