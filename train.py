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

# Motion-input experiment vs the main baseline (run 297). use_motion=True feeds
# [last action | velocity] through its own small encoder into the head, so the
# net can tell "approaching" from "stuck/reversing" — targets the oscillation
# failure (vibrating in place, worst at chokepoints but also in open space) that
# greedy argmax hits with no motion signal. depth-4 + random-tile, LayerNorm off.
# Watch Eval/chain_score (out of 5) vs 297.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, random_goal_tiles=True)

agent.train(episodes=900, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10)
