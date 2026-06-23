"""Hybrid terminal controller: drive with the learned navigator until within
terminal_radius of the goal, then steer ANALYTICALLY toward the known goal pixel
(pick the compass action whose direction best points at the goal). The learned Q's
greedy field orbits arbitrary goals ~30-64px out; an analytic compass step reduces
distance every step and converges to ~4px -- far inside the 31px trash tolerance --
bypassing the orbit. Realism-clean: dead-reckoning to a known commanded coordinate at
close range (line of sight), exactly what a real robot's final-approach servo does.

    python3 scripts/test_terminal_controller.py --checkpoint checkpoints/run314_q_model_best.pt \
        --readout softmax_rel --episodes 30
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401
from evaluate import load_q_model, process_observation
from chained_eval import _select_action, TRASH_REACH
from goal_geometry import distance, eval_step_budget
from motion import MotionState
from task_chain import resolve_goal
from planner.path_planner import plan_waypoints

_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _analytic_action(rx, ry, gx, gy):
    """Compass action whose direction best points at the goal (max dot product)."""
    vx, vy = gx - rx, gy - ry
    return max(range(8), key=lambda a: _DIRS[a][0] * vx + _DIRS[a][1] * vy)


def _drive(model, env, base, obs, goal, device, readout, temp, terminal_radius, budget,
           use_waypoints=False):
    r = base._robot
    ms = MotionState(base.action_space.n, getattr(model, "motion_window", 1))
    wps = plan_waypoints(base._map, (r.x, r.y), goal, stride=1) if use_waypoints else []
    wps = wps or [(float(goal[0]), float(goal[1]))]
    wp_i = 0
    best = distance(r.x, r.y, goal[0], goal[1])
    for _ in range(budget):
        d = distance(r.x, r.y, goal[0], goal[1])
        best = min(best, d)
        if d <= TRASH_REACH:
            return True, best
        if terminal_radius > 0 and d <= terminal_radius:
            a = _analytic_action(r.x, r.y, goal[0], goal[1])      # analytic final approach
        else:
            while wp_i < len(wps) - 1 and distance(r.x, r.y, wps[wp_i][0], wps[wp_i][1]) <= 24:
                wp_i += 1
            target = wps[wp_i]                                    # learned nav toward waypoint
            a = _select_action(model, obs, target, r, device, readout, temp, ms.vec(r.x, r.y))
        ms.commit(r.x, r.y, a)
        obs = process_observation(env.step(a)[0])
    best = min(best, distance(r.x, r.y, goal[0], goal[1]))
    return best <= TRASH_REACH, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/run314_q_model_best.pt")
    ap.add_argument("--head-norm", action="store_true")
    ap.add_argument("--readout", default="softmax_rel")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--radii", type=float, nargs="+", default=[0, 48, 64, 80])
    ap.add_argument("--waypoints", action="store_true",
                    help="also route the journey via A* waypoints (handles obstacle stalls)")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=2, head_layers=4, head_norm=args.head_norm,
                         use_motion=True)

    print(f"\ncheckpoint: {args.checkpoint} | readout={args.readout} | trash reach<="
          f"{TRASH_REACH:.0f}px | {args.episodes} scenes  (radius 0 = pure learned)")
    for tr in args.radii:
        n = args.episodes
        reached = 0
        for seed in range(n):
            raw, _ = env.reset(seed=seed)
            gx, gy = resolve_goal(base, "collect_trash")
            r = base._robot
            budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy)))) + 100
            ok, _ = _drive(model, env, base, process_observation(raw), (gx, gy),
                           device, args.readout, args.temp, tr, budget,
                           use_waypoints=args.waypoints)
            reached += ok
        tag = "+A* waypoints" if args.waypoints else ""
        print(f"  terminal_radius={tr:>4.0f}px {tag}:  {reached}/{n} = {100*reached/n:.0f}%")


if __name__ == "__main__":
    main()
