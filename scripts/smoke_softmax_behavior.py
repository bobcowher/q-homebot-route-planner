"""Integration smoke for the softmax-behavior training rollout policy.

Builds a real Agent both ways and exercises select_action with epsilon forced to
0 (so the random-exploration branch is bypassed and we test the exploit branch):
  - softmax_behavior=False -> deterministic argmax (same action every call)
  - softmax_behavior=True  -> stochastic softmax_rel (multiple distinct actions)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import homebot  # noqa: F401  (env registration)
from agent import Agent


def _make_agent(**kw):
    env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array",
                   action_mode="discrete", obs_resolution=(96, 96), n_trash=2,
                   max_steps=1000, map_name="default", goals=["collect_trash"],
                   random_start=True)
    return Agent(env=env, max_buffer_size=1000, goal_layers=2, head_layers=4,
                 use_motion=True, motion_window=1, random_goal_tiles=True, **kw)


def _obs_goal_motion(agent):
    raw, _ = agent.env.reset()
    obs = agent.process_observation(raw["observation"])
    goal = [10.0, 10.0, 400.0, 300.0]  # [robot_x, robot_y, goal_x, goal_y]
    from motion import MotionState
    ms = MotionState(agent.n_actions, agent.motion_window)
    return obs, goal, ms.vec(10.0, 10.0)


def main():
    # Deterministic (argmax) path
    a = _make_agent(softmax_behavior=False)
    a.epsilon = 0.0
    obs, goal, motion = _obs_goal_motion(a)
    greedy = {a.select_action(obs, goal, motion) for _ in range(30)}
    assert len(greedy) == 1, f"argmax must be deterministic, got {greedy}"
    assert 0 <= next(iter(greedy)) < a.n_actions

    # Stochastic (softmax_rel) path
    b = _make_agent(softmax_behavior=True, softmax_behavior_temp=0.1)
    b.epsilon = 0.0
    obs, goal, motion = _obs_goal_motion(b)
    actions = [b.select_action(obs, goal, motion) for _ in range(80)]
    assert all(0 <= x < b.n_actions for x in actions)
    distinct = set(actions)
    assert len(distinct) > 1, f"softmax_behavior must be stochastic, got {distinct}"

    print(f"OK | greedy action={next(iter(greedy))} | "
          f"softmax distinct actions={sorted(distinct)} over 80 samples")


if __name__ == "__main__":
    main()
