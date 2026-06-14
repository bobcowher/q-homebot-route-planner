"""Local smoke test: exercise Agent.train + greedy_eval end-to-end on a tiny run.

Uses the lowercase/capital env-id fallback so it runs locally (-V1) and would
also work remotely (-v1). Not a unit test — a fast wiring check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import homebot  # noqa: F401

from agent import Agent


def make_env():
    for env_id in ("HomeBot2D-Goal-v1", "HomeBot2D-Goal-V1"):
        try:
            return gym.make(
                env_id,
                render_mode="rgb_array",
                action_mode="discrete",
                obs_resolution=(96, 96),
                n_trash=2,
                max_steps=60,
                map_name="default",
                goals=["collect_trash"],
            )
        except gym.error.Error:
            continue
    raise RuntimeError("no env id registered")


env = make_env()
agent = Agent(env=env, max_buffer_size=2000)
agent.train(episodes=3, batch_size=8, eval_interval=1, eval_episodes=2)
print("SMOKE OK")
