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

# CURRICULUM + MOTION A/B arm vs run 305 (curriculum, no motion). Same locked
# curriculum as 303/305 (K=2 bootstrap 600, anneal 2->0 to ep1800, 200k buffer),
# but motion ON. Motion-input was a wash when tested ALONE (runs 299/301) — but
# that was under the old diet where broken routing capped the score and masked
# any anti-loop benefit. The curriculum cracked routing (fridge/door 0->70%+);
# the residual failure is the greedy action loop, which is exactly what motion
# targets. This is the fair re-test: does [last action | velocity] sand down the
# residual loops on top of the curriculum? Kill-or-cure for motion.
# Watch Eval/chain_score vs 305, and Eval/avg_success_steps (loop/directness).
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, random_goal_tiles=True)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
