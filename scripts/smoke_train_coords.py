"""End-to-end smoke of the coord-reframing training loop (a few episodes).

    conda run -n sac-homebot python scripts/smoke_train_coords.py

Exercises the full path that the model-only smoke can't: env reset, world_coords
goal build in select_action + train loop, HER relabel in EpisodeBuffer.send_to,
goal_dim=4 replay store/sample, a grad step, and one eval pass (greedy+softmax).
Catches goal-width mismatches before a real run.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent

env = gym.make(
    "HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
    obs_resolution=(96, 96), n_trash=2, max_steps=60, map_name="default",
    goals=["collect_trash"], random_start=True,
)

# Small buffer so can_sample() trips quickly and a grad step actually runs.
agent = Agent(env=env, max_buffer_size=5000, goal_layers=2, head_layers=2)
agent.min_epsilon = 1.0  # all-random: just exercise plumbing, fast

assert agent.goal_dim == 4
assert agent.memory.goal_memory.shape[1] == 4, agent.memory.goal_memory.shape

agent.train(episodes=3, batch_size=8, eval_interval=2, eval_episodes=2)
print("COORD TRAIN SMOKE OK")
