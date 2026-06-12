from dataclasses import dataclass
from typing import Callable
import random
import numpy as np
import torch


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        int
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_prev: np.ndarray  # robot pixel (x, y) at obs (before the step)
    achieved_next: np.ndarray  # robot pixel (x, y) at next_obs (after the step)


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    The network consumes relative goals (goal - robot position), so each
    transition needs the robot position at both s and s' — the relative goal
    changes within a transition. Rewards are still computed on absolutes.

    Usage:
        # each step:
        episode_buffer.store(obs, action, reward, next_obs, done,
                             achieved_prev, achieved_next)

        # end of episode:
        episode_buffer.send_to(replay_buffer, desired_goal, compute_reward)
        episode_buffer.clear()
    """

    K = 2  # hindsight goals per transition (future strategy)

    def __init__(self):
        self._transitions: list[Transition] = []

    def store(self, obs, action, reward, next_obs, done, achieved_prev, achieved_next):
        self._transitions.append(Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_prev=achieved_prev,
            achieved_next=achieved_next,
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
        """Write original transitions then K hindsight-relabeled copies to replay_buffer.

        Strategy: future — hindsight goals are sampled from achieved positions
        strictly after the current step. Last step is skipped (no future states).
        Goals are stored as relative vectors (absolute goal - robot position).
        """
        # Pass 1: original transitions (env reward, episode desired_goal)
        for t in self._transitions:
            replay_buffer.store_transition(
                t.obs, t.action, t.reward, t.next_obs, t.done,
                desired_goal - t.achieved_prev,
                desired_goal - t.achieved_next,
            )

        # Pass 2: hindsight transitions
        for i, t in enumerate(self._transitions):
            future = self._transitions[i + 1:]
            if not future:
                continue
            k = min(self.K, len(future))
            for hg_t in random.sample(future, k):
                hindsight_goal   = hg_t.achieved_next
                hindsight_reward = float(compute_reward(
                    t.achieved_next[np.newaxis],
                    hindsight_goal[np.newaxis],
                    {},
                )[0])
                # Success terminates in this env (reward > 0.5 -> terminated), so a
                # relabeled success must be terminal too — otherwise targets bootstrap
                # past the goal and inflate Q toward 1/(1-gamma) in hindsight data.
                hindsight_done = hindsight_reward > 0.5
                replay_buffer.store_transition(
                    t.obs, t.action, hindsight_reward, t.next_obs, hindsight_done,
                    hindsight_goal - t.achieved_prev,
                    hindsight_goal - t.achieved_next,
                )
