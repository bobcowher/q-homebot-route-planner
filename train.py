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

# Main baseline: locked depth-4 navigator (goal_layers=2/head=4), random-tile
# goals (whole-map go-to objective). LayerNorm left OFF — its A/B (run 296) only
# redistributed competence (fridge up, recliner down), net-neutral per-goal and
# worse on the chain score. The real metric (Eval/chain_score, out of 5) is now
# logged every 10 episodes. Episodes 900: eval plateaus by ~ep700-750.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              random_goal_tiles=True)

agent.train(episodes=900, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10)
