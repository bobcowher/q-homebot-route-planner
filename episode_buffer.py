from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch


@dataclass
class Transition:
    obs:           torch.Tensor
    action:        int
    reward:        float
    next_obs:      torch.Tensor
    done:          bool
    achieved_goal: np.ndarray  # robot pixel (x, y) at this step


class EpisodeBuffer:
    """Caches one episode's transitions for HER relabeling.

    Usage:
        # each step:
        episode_buffer.store(obs, action, reward, next_obs, done, achieved_goal)

        # end of episode:
        episode_buffer.flush_to(replay_buffer, desired_goal)
        episode_buffer.clear()
    """

    def __init__(self):
        self._transitions: list[Transition] = []

    def store(self, obs, action, reward, next_obs, done, achieved_goal):
        self._transitions.append(Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            achieved_goal=achieved_goal,
        ))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions.clear()

    def flush_to(
        self,
        replay_buffer,
        desired_goal: np.ndarray,
        compute_reward: Callable,
    ) -> None:
        """Store original transitions, then hindsight-relabeled transitions.

        TODO: implement HER relabeling.
              1. Store each original transition as-is (reward from env).
              2. For each step i, sample a hindsight goal from achieved_goals[i+1:]
                 (strategy='future').
              3. Relabeled reward = compute_reward(t.next_obs achieved_goal,
                 hindsight_goal, {})  — batched, returns float32 0/1.
              4. Store the relabeled transition.
        """
        # --- original transitions ---
        for t in self._transitions:
            replay_buffer.store_transition(t.obs, t.action, t.reward, t.next_obs, t.done)

        # --- hindsight transitions (TODO) ---
        # import random
        # for i, t in enumerate(self._transitions):
        #     future = self._transitions[i + 1:]
        #     if not future:
        #         continue
        #     hindsight_goal = random.choice(future).achieved_goal
        #     hindsight_reward = float(compute_reward(t.achieved_goal, hindsight_goal, {}))
        #     replay_buffer.store_transition(
        #         t.obs, t.action, hindsight_reward, t.next_obs, t.done
        #     )
        _ = desired_goal  # will be used by hindsight block above
        _ = compute_reward
