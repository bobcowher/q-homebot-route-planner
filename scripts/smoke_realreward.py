"""Headless smoke for the real-reward reach fix: build the train env (real
TaskManager reward + reward-fires termination, RELABEL_RADIUS HER proxy) and the
champion-314 Agent, run a few short episodes through rollout -> HER relabel -> train
step. Confirms the reward path, HER send_to, and a gradient step all run without
crashing, and that a real +1 reward actually fires when the robot reaches the trash.
"""
import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401
from agent import Agent
from homebot.goals import RELABEL_RADIUS, GOAL_THRESHOLD

print(f"RELABEL_RADIUS={RELABEL_RADIUS} (proxy)  GOAL_THRESHOLD={GOAL_THRESHOLD} (fixture)")

env = gym.make("HomeBot2D-Goal-V1", render_mode="rgb_array", action_mode="discrete",
               obs_resolution=(96, 96), n_trash=1, max_steps=60,
               map_name="default", goals=["collect_trash"], random_start=True)

# Real reward fires only at the true 31px trash radius: teleport onto the trash and
# step, confirm +1 from the TaskManager (not the geometric proxy).
base = env.unwrapped
env.reset(seed=0)
gx, gy = base._desired_goal
base._robot.x, base._robot.y = float(gx), float(gy)
_, reward, terminated, _, _ = env.step(0)
print(f"on-trash step -> reward={reward} terminated={terminated}  (expect 1.0 / True)")
assert reward == 1.0 and terminated, "real task reward did not fire at the trash"

agent = Agent(env=env, max_buffer_size=5000, goal_layers=2, head_layers=4,
              use_motion=True, motion_window=1)
agent.train(episodes=3, batch_size=16, eval_interval=100, eval_episodes=2,
            chain_eval_interval=100, her_anneal_start=None)
print("OK | real-reward env + HER rollout + train step ran clean")
