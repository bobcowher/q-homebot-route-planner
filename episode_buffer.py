from dataclasses import dataclass
from typing import Callable
import random
import numpy as np
import torch

from goal_geometry import world_coords


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        int
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_prev: np.ndarray  # robot pixel (x, y) at obs (before the step)
    achieved_next: np.ndarray  # robot pixel (x, y) at next_obs (after the step)
    heading_prev:  float       # robot.angle (radians) before the step
    heading_next:  float       # robot.angle (radians) after the step
    motion_prev:   np.ndarray | None = None  # motion feature at s  (state-intrinsic)
    motion_next:   np.ndarray | None = None  # motion feature at s' (state-intrinsic)


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    Coordinate reframing: the network consumes the goal as absolute coords
    [robot_x, robot_y, goal_x, goal_y]. The robot-pose half changes within a
    transition as the robot moves, so each transition stores robot position at
    both s and s' (heading retained for compatibility but unused by the coord
    rep). Rewards still computed from absolute positions.
    """

    K = 2  # hindsight goals per transition (future strategy)

    def __init__(self):
        self._transitions: list[Transition] = []

    def store(self, obs, action, reward, next_obs, done,
              achieved_prev, achieved_next,
              heading_prev: float = 0.0, heading_next: float = 0.0,
              motion_prev=None, motion_next=None):
        self._transitions.append(Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_prev=achieved_prev,
            achieved_next=achieved_next,
            heading_prev=float(heading_prev),
            heading_next=float(heading_next),
            motion_prev=motion_prev,
            motion_next=motion_next,
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def send_to(
        self,
        replay_buffer,
        desired_goal: np.ndarray,
        compute_reward: Callable,
    ) -> None:
        """Write original transitions then K hindsight-relabeled copies.

        Strategy: future. Goals stored as absolute coords
        [robot_x, robot_y, goal_x, goal_y]; the robot-pose half is the pose at
        that transition, the goal half is the (desired or hindsight) target.
        """
        dg = desired_goal  # absolute map-px (x, y)

        # Pass 1: original transitions (env reward, episode desired_goal)
        for t in self._transitions:
            goal_at_s  = world_coords(t.achieved_prev[0], t.achieved_prev[1],
                                      dg[0], dg[1])
            goal_at_sp = world_coords(t.achieved_next[0], t.achieved_next[1],
                                      dg[0], dg[1])
            replay_buffer.store_transition(
                t.obs, t.action, t.reward, t.next_obs, t.done,
                goal_at_s, goal_at_sp,
                motion=t.motion_prev, next_motion=t.motion_next,
            )

        # Pass 2: hindsight transitions
        for i, t in enumerate(self._transitions):
            future = self._transitions[i + 1:]
            if not future:
                continue
            k = min(self.K, len(future))
            for hg_t in random.sample(future, k):
                hindsight_goal   = hg_t.achieved_next  # absolute map-px
                hindsight_reward = float(compute_reward(
                    t.achieved_next[np.newaxis],
                    hindsight_goal[np.newaxis],
                    {},
                )[0])
                # Success terminates in this env (reward > 0.5 -> terminated), so a
                # relabeled success must be terminal too — otherwise targets bootstrap
                # past the goal and inflate Q toward 1/(1-gamma) in hindsight data.
                hindsight_done = hindsight_reward > 0.5
                hs_goal_at_s  = world_coords(t.achieved_prev[0], t.achieved_prev[1],
                                             hindsight_goal[0], hindsight_goal[1])
                hs_goal_at_sp = world_coords(t.achieved_next[0], t.achieved_next[1],
                                             hindsight_goal[0], hindsight_goal[1])
                replay_buffer.store_transition(
                    t.obs, t.action, hindsight_reward, t.next_obs, hindsight_done,
                    hs_goal_at_s, hs_goal_at_sp,
                    motion=t.motion_prev, next_motion=t.motion_next,
                )
