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

# LayerNorm A/B vs run 295 (depth-4, random-tile, no norm). head_norm=True adds
# LayerNorm after each head Linear — the value-RL-safe head regularizer. Target:
# close 295's train/eval gap (train EMA 0.88 vs greedy eval 0.51) and stabilize.
# Episodes cut 1200->900: 295's eval plateaued by ~ep700-750, the rest was noise.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              head_norm=True, random_goal_tiles=True)

agent.train(episodes=900, batch_size=64, eval_interval=50, eval_episodes=20)
