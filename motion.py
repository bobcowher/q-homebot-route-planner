"""Previous-motion features for the Q-net (anti-oscillation).

The single-frame policy has no sense of its own motion, so at ambiguous
(state, goal) configs the greedy argmax flips between opposing actions and the
robot vibrates in place (worst at chokepoints, but also in open space). We hand
it the missing temporal state: the last action (intent) + the realized velocity
(outcome — ~0 means "I'm blocked/stuck", exactly the freak-out state made
observable). The Q-net routes this through its own small encoder.

One step of velocity only exposes the STATIONARY freak-out (velocity ~0). It is
blind to the *moving* limit cycle (spinning): a robot circling a 2-cycle has a
full-size per-step velocity every step, so it looks like healthy motion. To make
spinning observable we add a windowed term: net displacement over the last
`window` poses. During a cycle the per-step velocity stays large while the
windowed net collapses to ~0 — a distinct input the net can finally condition on
to break the loop. On a real robot this is just odometry over a window (no
ground-truth or reward hack). `window<=1` disables it and reproduces the original
[one-hot | velocity] vector exactly (keeps old checkpoints loadable).

State-intrinsic, so HER never relabels it (unlike the goal).
"""
from collections import deque

import numpy as np

from goal_geometry import ROBOT_STEP_PX

# Default trailing window for the net-displacement term. Matches the spin metric
# (scripts/spin_metric.py) so the feature sees the same horizon we score against.
MOTION_WINDOW = 8


def motion_dim(action_dim: int, window: int = 1) -> int:
    """Motion vector width: one-hot last action + (dx, dy), plus (net_dx, net_dy)
    over the window when window > 1."""
    return action_dim + 2 + (2 if window > 1 else 0)


def make_motion(action_dim, last_action, dx, dy, net_dx=0.0, net_dy=0.0,
                step=ROBOT_STEP_PX, window=1):
    """[ one-hot(last_action) | dx/step | dy/step | net_dx/(W*step) | net_dy/(W*step) ].

    last_action None/<0 -> zeros for the one-hot (episode start). Velocity is
    normalized by the discrete step so it sits in ~[-1, 1]. The windowed net is
    normalized by window*step so a straight run reads ~1 and a closed cycle ~0.
    When window<=1 the net dims are omitted (original n+2 vector)."""
    m = np.zeros(motion_dim(action_dim, window), dtype=np.float32)
    if last_action is not None and last_action >= 0:
        m[int(last_action)] = 1.0
    m[action_dim] = dx / step
    m[action_dim + 1] = dy / step
    if window > 1:
        m[action_dim + 2] = net_dx / (window * step)
        m[action_dim + 3] = net_dy / (window * step)
    return m


class MotionState:
    """Per-rollout tracker. Usage each step, at robot pose (x, y):

        motion = ms.vec(x, y)          # velocity into this state + last action
        action = argmax Q(obs, goal, motion)
        ms.commit(x, y, action)        # record pose+action, then env.step(action)

    Next step's vec(x', y') then sees dx = x' - x. With window>1 it also reports
    the net displacement from the pose `window` steps back, so a 2-cycle (moving
    but not progressing) becomes a distinct input. Carries across chained legs
    (one MotionState per episode); reset() at episode boundaries.
    """

    def __init__(self, action_dim: int, window: int = 1):
        self.adim = action_dim
        self.window = window
        self.reset()

    def reset(self):
        self.last_action = None
        self.prev = None  # pose before the current state
        # last `window` committed poses; history[0] is ~window steps back.
        self.history = deque(maxlen=max(1, self.window))

    def vec(self, x, y):
        if self.prev is None:
            dx = dy = 0.0
        else:
            dx, dy = x - self.prev[0], y - self.prev[1]
        if self.window > 1 and self.history:
            ox, oy = self.history[0]
            net_dx, net_dy = x - ox, y - oy
        else:
            net_dx = net_dy = 0.0
        return make_motion(self.adim, self.last_action, dx, dy,
                           net_dx, net_dy, window=self.window)

    def commit(self, x, y, action):
        self.history.append((x, y))
        self.prev = (x, y)
        self.last_action = action
