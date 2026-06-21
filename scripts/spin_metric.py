"""Baseline metric for the SPINNING failure mode (limit cycles / circling).

Unlike diagnose_stuck (which only inspects timed-out legs), this scores EVERY
leg -- including ones that succeed after spinning, which is the behavior we
actually see: the robot loops, then breaks out. A step counts as "spinning" when,
over a trailing window, the robot moved a lot but got nowhere (high path length,
low net displacement). That isolates true loops from wall-stick (little motion)
and from clean progress (net ~ path).

Reports, per readout (greedy AND the deployed softmax_rel):
  - mean spin fraction: share of all nav steps spent moving-without-progress
  - legs that spun: fraction of legs with a meaningful spin
  - of REACHED legs, the share that spun first ("broke out") -- the live symptom
  - path inflation: total path / straight-line ideal

    python3 scripts/spin_metric.py --checkpoint checkpoints/run314_q_model_best.pt \
        --goal-layers 2 --head-layers 4 --use-motion --episodes 20
"""
import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401  (env registration)
from homebot.goals import GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import distance, eval_step_budget, ROBOT_STEP_PX, spin_fraction
from motion import MotionState
from chained_eval import _select_action, REACH_OVERRIDE
from task_chain import DEFAULT_CHAIN, resolve_goal


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# spin_fraction now lives in goal_geometry (shared with the in-train chain eval);
# re-exported above so callers and the test keep importing it from here.


def leg_positions(model, env, base, obs, goal_xy, budget, device, readout, temp, ms, reach):
    """Drive one leg; return (reached, positions, obs). positions is the per-step
    (x, y) trace used for the spin computation."""
    robot = base._robot
    positions = [(robot.x, robot.y)]
    reached = False
    for _ in range(budget):
        motion = ms.vec(robot.x, robot.y)
        action = _select_action(model, obs, goal_xy, robot, device, readout, temp, motion)
        ms.commit(robot.x, robot.y, action)
        obs = process_observation(env.step(action)[0])
        positions.append((robot.x, robot.y))
        if distance(robot.x, robot.y, goal_xy[0], goal_xy[1]) <= reach:
            reached = True
            break
    return reached, positions, obs


def _run(model, env, base, readout, temp, episodes, seed, window, move_min, net_max):
    legs = []  # (name, reached, spin_frac, path, straight)
    for ep in range(episodes):
        obs = process_observation(env.reset(seed=seed + ep)[0])
        r = base._robot
        ms = MotionState(env.action_space.n, getattr(model, "motion_window", 1))
        targets = [(name, resolve_goal(base, name)) for name in DEFAULT_CHAIN]
        for name, (gx, gy) in targets:
            start = (r.x, r.y)
            budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
            reach = REACH_OVERRIDE.get(name, GOAL_THRESHOLD)
            reached, pos, obs = leg_positions(model, env, base, obs, (gx, gy),
                                              budget, "cuda:0" if torch.cuda.is_available()
                                              else "cpu", readout, temp, ms, reach)
            sf = spin_fraction(pos, window, move_min, net_max)
            path = sum(_d(pos[i - 1], pos[i]) for i in range(1, len(pos)))
            straight = _d(start, pos[-1])
            legs.append((name, reached, sf, path, straight))
    return legs


def _report(readout, legs, spin_leg_thresh):
    n = len(legs)
    mean_sf = sum(l[2] for l in legs) / n if n else 0.0
    spun = [l for l in legs if l[2] >= spin_leg_thresh]
    reached = [l for l in legs if l[1]]
    reached_spun = [l for l in reached if l[2] >= spin_leg_thresh]
    total_path = sum(l[3] for l in legs)
    total_straight = sum(l[4] for l in legs)
    inflation = (total_path / total_straight) if total_straight else float("nan")

    print(f"\n=== spin metric | readout={readout} | {n} legs ===")
    print(f"  mean spin fraction:        {100 * mean_sf:.1f}%  "
          f"(share of nav steps moving-without-progress)")
    print(f"  legs that spun (>={int(100*spin_leg_thresh)}%):     "
          f"{len(spun)}/{n} = {100 * len(spun) / n:.0f}%")
    print(f"  reached legs that spun:    "
          f"{len(reached_spun)}/{len(reached)} = "
          f"{(100 * len(reached_spun) / len(reached)) if reached else 0:.0f}%  "
          f"(spun, then broke out)")
    print(f"  path inflation:            {inflation:.2f}x straight-line")
    return mean_sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run314_q_model_best.pt")
    p.add_argument("--goal-layers", type=int, default=2)
    p.add_argument("--head-layers", type=int, default=4)
    p.add_argument("--head-norm", action="store_true")
    p.add_argument("--use-motion", action="store_true")
    p.add_argument("--motion-window", type=int, default=1,
                   help="windowed net-displacement horizon the checkpoint was "
                        "trained with (1 = original velocity-only motion)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--readouts", nargs="+", default=["greedy", "softmax_rel"],
                   choices=["greedy", "softmax", "softmax_rel"])
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--window", type=int, default=8,
                   help="trailing steps over which to judge net progress")
    p.add_argument("--spin-leg-thresh", type=float, default=0.1,
                   help="a leg 'spun' if this fraction of its steps were spinning")
    args = p.parse_args()

    # Thresholds in ROBOT_STEP_PX units so they track the env step size.
    move_min = 0.5 * args.window * ROBOT_STEP_PX  # really moved over the window
    net_max = 2.0 * ROBOT_STEP_PX                 # but ended ~where it started

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=args.goal_layers, head_layers=args.head_layers,
                         head_norm=args.head_norm, use_motion=args.use_motion,
                         motion_window=args.motion_window)

    print(f"checkpoint: {args.checkpoint} | window={args.window} "
          f"move_min={move_min:.1f}px net_max={net_max:.1f}px")
    summary = {}
    for readout in args.readouts:
        legs = _run(model, env, base, readout, args.temp, args.episodes, args.seed,
                    args.window, move_min, net_max)
        summary[readout] = _report(readout, legs, args.spin_leg_thresh)

    print(f"\n=== spin summary (mean spin fraction) ===")
    for readout, sf in summary.items():
        print(f"  {readout:<12} {100 * sf:.1f}%")


if __name__ == "__main__":
    main()
