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

# HER-curriculum experiment vs the main baseline. NO motion (use_motion=False) so
# this isolates the curriculum effect: bootstrap on heavy hindsight (K=2) for 600
# episodes, then anneal hindsight 2 -> 0 over 600->1200 so the buffer's marginal
# inflow leans toward real (un-relabeled) data — a fine-tune off the relabeled
# diet. HER relabels wandering failures as successes, which may train in the
# oscillation; leaning off it late should sharpen toward direct paths. Buffer
# stays 200k (a smaller buffer is its own separate test). depth-4, random-tile.
# Watch Train/hindsight_k (the schedule), Eval/avg_success_steps (path
# directness = the oscillation signal), and per-fixture reach for regressions.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=False, random_goal_tiles=True)

agent.train(episodes=1200, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
