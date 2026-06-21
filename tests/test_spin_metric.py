"""spin_fraction: detect moving-without-progress windows (limit cycles)."""
from scripts.spin_metric import spin_fraction

# Explicit thresholds (step size 1.0): a window 'spins' if it walked >=3 of path
# but ended within 1.5 of where it began 4 steps earlier.
W, MOVE_MIN, NET_MAX = 4, 3.0, 1.5


def test_straight_line_is_not_spinning():
    pos = [(i, 0.0) for i in range(20)]  # steady progress, step 1.0
    assert spin_fraction(pos, W, MOVE_MIN, NET_MAX) == 0.0


def test_back_and_forth_is_fully_spinning():
    pos = [(i % 2, 0.0) for i in range(20)]  # 0,1,0,1,... a 2-cycle
    assert spin_fraction(pos, W, MOVE_MIN, NET_MAX) == 1.0


def test_wall_stick_is_not_counted_as_spin():
    pos = [(0.0, 0.0)] * 20  # no motion at all -> path < MOVE_MIN
    assert spin_fraction(pos, W, MOVE_MIN, NET_MAX) == 0.0


def test_trace_shorter_than_window_is_zero():
    assert spin_fraction([(0.0, 0.0), (1.0, 0.0)], W, MOVE_MIN, NET_MAX) == 0.0


def test_spin_then_escape_is_partial():
    # circles in place for a while (2-cycle), then walks straight away
    spinning = [(i % 2, 0.0) for i in range(12)]
    escaping = [(float(i), 0.0) for i in range(2, 14)]  # monotonic progress
    frac = spin_fraction(spinning + escaping, W, MOVE_MIN, NET_MAX)
    assert 0.0 < frac < 1.0
