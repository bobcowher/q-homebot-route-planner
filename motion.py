"""Previous-motion features for the Q-net (anti-oscillation).

The single-frame policy has no sense of its own motion, so at ambiguous
(state, goal) configs the greedy argmax flips between opposing actions and the
robot vibrates in place (worst at chokepoints, but also in open space). We hand
it the missing temporal state: the last action (intent) + the realized velocity
(outcome — ~0 means "I'm blocked/stuck", exactly the freak-out state made
observable). The Q-net routes this through its own small encoder.

State-intrinsic, so HER never relabels it (unlike the goal).
"""
import numpy as np

from goal_geometry import ROBOT_STEP_PX


def motion_dim(action_dim: int) -> int:
    """Motion vector width: one-hot last action + (dx, dy)."""
    return action_dim + 2


def make_motion(action_dim, last_action, dx, dy, step=ROBOT_STEP_PX):
    """[ one-hot(last_action) | dx/step | dy/step ]. last_action None/<0 -> zeros
    (episode start). Velocity normalized by the discrete step so it sits in ~[-1,1]."""
    m = np.zeros(action_dim + 2, dtype=np.float32)
    if last_action is not None and last_action >= 0:
        m[int(last_action)] = 1.0
    m[action_dim] = dx / step
    m[action_dim + 1] = dy / step
    return m


class MotionState:
    """Per-rollout tracker. Usage each step, at robot pose (x, y):

        motion = ms.vec(x, y)          # velocity into this state + last action
        action = argmax Q(obs, goal, motion)
        ms.commit(x, y, action)        # record pose+action, then env.step(action)

    Next step's vec(x', y') then sees dx = x' - x. Carries across chained legs
    (one MotionState per episode); reset() at episode boundaries.
    """

    def __init__(self, action_dim: int):
        self.adim = action_dim
        self.reset()

    def reset(self):
        self.last_action = None
        self.prev = None  # pose before the current state

    def vec(self, x, y):
        if self.prev is None:
            dx = dy = 0.0
        else:
            dx, dy = x - self.prev[0], y - self.prev[1]
        return make_motion(self.adim, self.last_action, dx, dy)

    def commit(self, x, y, action):
        self.prev = (x, y)
        self.last_action = action
