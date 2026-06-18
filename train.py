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

# LOCKED CURRICULUM (run 303), extended to find the plateau. The HER curriculum
# cracked cross-room routing: bootstrap on heavy hindsight (K=2) for 600 episodes,
# then anneal hindsight 2 -> 0 over the rest so the marginal inflow leans toward
# real (un-relabeled) data, while the 200k buffer keeps relabeled goal DIVERSITY
# as ballast (smaller buffers choke that diversity and underperform — run 304).
# 303 (1200ep) hit chain 3.85/5 greedy with fridge/door 0->70%+, and chain_score
# was still rising at the end. This run pushes to 1800 to find where it tops out.
# NOTE: anneal end scales with episodes, so k reaches ~0 at ep1800 here (a gentler
# taper than 303's 600->1200). depth-4, random-tile, no motion.
# Watch Eval/chain_score (does it keep climbing past 1200?), Train/hindsight_k.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=False, random_goal_tiles=True)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
