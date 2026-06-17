"""Integration smoke for the motion input: exercises rollout+store (motion in
the episode/replay buffer), train_step (motion sampled + fed to both Q-nets),
and all three eval paths (greedy/softmax/chain) with motion.

    conda run -n sac-homebot python scripts/smoke_motion.py
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
agent = Agent(env=env, max_buffer_size=5000, goal_layers=2, head_layers=4,
              use_motion=True, random_goal_tiles=True)

assert agent.use_motion and agent.motion_dim == agent.n_actions + 2
assert agent.q_model.use_motion and agent.q_model.motion_encoder is not None
assert agent.memory.use_motion

# Full integration: a couple short episodes -> store (with motion) + train_step
# (samples motion) + greedy/softmax/chain eval (all build motion vectors).
agent.train(episodes=2, batch_size=8, eval_interval=1, eval_episodes=1,
            chain_eval_interval=1)

# Buffer actually carries motion of the right width.
*_, motions, next_motions = agent.memory.sample_buffer(4)
assert motions is not None and motions.shape[1] == agent.motion_dim
assert next_motions is not None and next_motions.shape[1] == agent.motion_dim
print(f"motion buffer width={motions.shape[1]} OK")
print("MOTION SMOKE OK")
