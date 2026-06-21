from agent import Agent
import gymnasium as gym
import homebot

env = gym.make(
    "HomeBot2D-Goal-V1",
    render_mode="rgb_array",
    action_mode="discrete",
    obs_resolution=(96, 96),
    n_trash=2,
    max_steps=1000,
    map_name="default",
    goals=["collect_trash"],
    random_start=True,   # env owns spawn now (uniform valid tile, >=60px from goals)
)

# WINDOWED MOTION vs champion 314 (depth-4 + velocity-only motion). 314 still
# spins (greedy 11.4% / softmax_rel 5.7% spin fraction, scripts/spin_metric.py).
# Diagnosis: one-step velocity only exposes the STATIONARY freak-out (velocity~0).
# A *moving* limit cycle has full per-step velocity, so it looks like healthy
# motion — invisible to [last action | velocity]. motion_window=8 adds the net
# displacement over the last 8 poses (matches the spin-metric horizon): during a
# cycle the windowed net collapses to ~0 while velocity stays large, a distinct
# input the net can finally condition on to break the loop. Realism-clean: it's
# odometry over a window, not a reward/ground-truth hack.
# Judge: honest chain (bar 4.50/55%, chained_eval.py) + spin_metric (beat 5.7%
# deployed / 11.4% greedy), both run with --motion-window 8.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=8, random_goal_tiles=True)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
