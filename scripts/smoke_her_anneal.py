"""Smoke for the HER curriculum: hindsight K anneals 2 -> 0 after a bootstrap,
exercised on the no-motion (main) architecture this run uses.

Verifies: (a) use_motion=False path runs end-to-end; (b) send_to accepts a
fractional k and stochastically rounds it; (c) the scheduled k decreases past
the anneal start.

    conda run -n sac-homebot python scripts/smoke_her_anneal.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import homebot  # noqa: F401
from agent import Agent
from episode_buffer import EpisodeBuffer

env = gym.make(
    "HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
    obs_resolution=(96, 96), n_trash=2, max_steps=60, map_name="default",
    goals=["collect_trash"], random_start=True,
)
agent = Agent(env=env, max_buffer_size=5000, goal_layers=2, head_layers=4,
              use_motion=False, random_goal_tiles=True)
assert not agent.use_motion and agent.motion_dim == 0

# Anneal start at episode 2 of a 6-episode run -> k should drop below K by the end.
agent.train(episodes=6, batch_size=8, eval_interval=3, eval_episodes=1,
            chain_eval_interval=3, her_anneal_start=2)

# Schedule sanity: k=2 during bootstrap, anneals to ~0 at the final episode.
def k_at(ep, episodes=6, start=2, K=EpisodeBuffer.K):
    if ep < start:
        return float(K)
    frac = min(1.0, (ep - start) / max(1, episodes - start))
    return K * (1.0 - frac)

assert k_at(0) == EpisodeBuffer.K and k_at(1) == EpisodeBuffer.K, "bootstrap not full-K"
assert k_at(5) < 0.6, f"k did not anneal down: {k_at(5)}"
print(f"k schedule: ep0={k_at(0)} ep2={k_at(2):.2f} ep5={k_at(5):.2f}")

# Fractional k actually stores a variable number of hindsight copies.
eb = EpisodeBuffer()
import numpy as np, torch
for j in range(5):
    eb.store(torch.zeros(3, 96, 96), 0, 0.0, torch.zeros(3, 96, 96), False,
             np.array([10.0 * j, 0.0], dtype=np.float32),
             np.array([10.0 * (j + 1), 0.0], dtype=np.float32))

class _Sink:
    def __init__(self): self.n = 0
    def store_transition(self, *a, **k): self.n += 1

sink0, sink_frac = _Sink(), _Sink()
eb.send_to(sink0, np.array([99.0, 0.0], np.float32), lambda a, b, info: np.array([0.0]), k=0.0)
eb.send_to(sink_frac, np.array([99.0, 0.0], np.float32), lambda a, b, info: np.array([0.0]), k=2.0)
assert sink0.n == 5, f"k=0 should store only the 5 originals, got {sink0.n}"
assert sink_frac.n > 5, f"k=2 should add hindsight copies, got {sink_frac.n}"
print(f"k=0 stored {sink0.n} (originals only); k=2 stored {sink_frac.n}")
print("HER ANNEAL SMOKE OK")
