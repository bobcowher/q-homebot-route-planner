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

# SOFTMAX BEHAVIOR POLICY vs champion 314. 314 trains epsilon-greedy but deploys
# softmax_rel (the anti-spin readout: 5.7% spin vs 11.4% greedy). So the Q-function
# never collected data under the policy we ship — a train/deploy mismatch. This run
# uses softmax_rel AS the rollout exploit policy (softmax_behavior=True, temp=0.1),
# so the agent learns to act well under its own deploy-time sampling. Hard-Q targets
# are untouched (unlike soft_q, which flattened Q — see soft_q verdict). Two prior
# anti-spin attempts failed by FIGHTING softmax_rel (windowed-input run 318,
# Q-penalty run 320); this one IS softmax_rel, so it can't cannibalize it.
# Config = champion 314: depth-4, velocity-only motion (motion_window=1).
# Judge: honest chain (chained_eval.py --readouts greedy softmax_rel --temp 0.1)
# + spin_metric, vs 314 best (4.30/38% chain, 5.8% deploy spin). Watch whether
# GREEDY closes on softmax_rel (the mismatch closing) without losing chain.
agent = Agent(env=env, max_buffer_size=200000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=1, random_goal_tiles=True,
              softmax_behavior=True, softmax_behavior_temp=0.1)

agent.train(episodes=1800, batch_size=64, eval_interval=50, eval_episodes=20,
            chain_eval_interval=10, her_anneal_start=600)
