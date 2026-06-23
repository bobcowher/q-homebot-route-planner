"""Does an A* waypoint layer rescue the navigator's long-tail failures? For each scene,
drive to the trash pile two ways and compare reach@31px:
  - DIRECT: one run_leg straight to the trash pixel (today's behavior)
  - WAYPOINT: A* short hops (planner/path_planner) through the navigator, pose
    persisting across hops, exact trash pixel as the final hop
If WAYPOINT >> DIRECT, the navigator was fine -- the missing piece is planning, and
the hierarchy fix is validated (no retraining).

    python3 scripts/test_waypoint_nav.py --checkpoint checkpoints/run314_q_model_best.pt \
        --readout softmax_rel --episodes 30 --stride 2
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401
from evaluate import load_q_model, process_observation
from chained_eval import run_leg, TRASH_REACH
from goal_geometry import distance, eval_step_budget
from motion import MotionState
from task_chain import resolve_goal
from planner.path_planner import plan_waypoints


def _drive_direct(model, env, base, obs, goal, device, readout, temp):
    ms = MotionState(base.action_space.n, getattr(model, "motion_window", 1))
    r = base._robot
    budget = max(1, int(eval_step_budget(distance(r.x, r.y, goal[0], goal[1]))))
    reached, _, _, pos = run_leg(model, env, base, obs, goal, budget, device,
                                 readout, temp, ms, TRASH_REACH)
    md = min(distance(x, y, goal[0], goal[1]) for x, y in pos)
    return reached, md


def _drive_waypoints(model, env, base, obs, goal, device, readout, temp, stride):
    wps = plan_waypoints(base._map, (base._robot.x, base._robot.y), goal, stride=stride)
    if not wps:                                  # no path -> fall back to direct
        return _drive_direct(model, env, base, obs, goal, device, readout, temp)
    ms = MotionState(base.action_space.n, getattr(model, "motion_window", 1))
    r = base._robot
    best_md = distance(r.x, r.y, goal[0], goal[1])
    reached = False
    for i, wp in enumerate(wps):
        final = (i == len(wps) - 1)
        # intermediate hops just need to get near the waypoint; final hop uses the
        # real trash tolerance.
        reach = TRASH_REACH if final else 24.0
        budget = max(1, int(eval_step_budget(distance(r.x, r.y, wp[0], wp[1]))))
        _, _, obs, pos = run_leg(model, env, base, obs, wp, budget, device,
                                 readout, temp, ms, reach)
        best_md = min(best_md, min(distance(x, y, goal[0], goal[1]) for x, y in pos))
        if best_md <= TRASH_REACH:
            reached = True
            break
    return reached, best_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/run314_q_model_best.pt")
    ap.add_argument("--head-norm", action="store_true")
    ap.add_argument("--readout", default="softmax_rel")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="waypoint every N tiles")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=2, head_layers=4, head_norm=args.head_norm,
                         use_motion=True)

    d_reach = w_reach = 0
    for seed in range(args.episodes):
        raw, _ = env.reset(seed=seed)
        gx, gy = resolve_goal(base, "collect_trash")
        dr, _ = _drive_direct(model, env, base, process_observation(raw),
                              (gx, gy), device, args.readout, args.temp)
        raw, _ = env.reset(seed=seed)            # same scene, fresh start
        wr, _ = _drive_waypoints(model, env, base, process_observation(raw),
                                 (gx, gy), device, args.readout, args.temp, args.stride)
        d_reach += dr
        w_reach += wr

    n = args.episodes
    print(f"\ncheckpoint: {args.checkpoint} | readout={args.readout} | trash reach<="
          f"{TRASH_REACH:.0f}px | stride={args.stride} | {n} scenes")
    print(f"  DIRECT (today):     {d_reach}/{n} = {100*d_reach/n:.0f}%")
    print(f"  A* WAYPOINTS:       {w_reach}/{n} = {100*w_reach/n:.0f}%")


if __name__ == "__main__":
    main()
