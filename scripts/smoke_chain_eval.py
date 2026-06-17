"""Smoke-test Agent.chain_eval (the in-training task-score metric).

    conda run -n sac-homebot python scripts/smoke_chain_eval.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from task_chain import DEFAULT_CHAIN

env = gym.make(
    "HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
    obs_resolution=(96, 96), n_trash=2, max_steps=1000, map_name="default",
    goals=["collect_trash"], random_start=True,
)
agent = Agent(env=env, max_buffer_size=1000, goal_layers=2, head_layers=4,
              random_goal_tiles=True)

score, full = agent.chain_eval(n_episodes=2)
assert 0.0 <= score <= len(DEFAULT_CHAIN), f"score {score} out of range"
assert 0.0 <= full <= 1.0, f"full {full} out of range"
assert agent._chain_env is not None, "chain env not lazily built"
print(f"chain_eval ran: score={score:.2f}/{len(DEFAULT_CHAIN)} full={full:.2f} (random net) OK")
print("CHAIN_EVAL SMOKE OK")
