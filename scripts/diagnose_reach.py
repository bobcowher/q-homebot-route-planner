"""Is the collect_trash failure a GENERAL terminal-precision floor, or trash-specific?

Drive the navigator to each named goal at a CONFIGURABLE reach tolerance and report
reach rate + the min-distance distribution. The decisive comparison: fixtures at their
normal 79px vs fixtures at trash's 31px. If fixtures also collapse at 31px, the
"trash problem" is really a general convergence/precision floor (architecture), only
visible on trash because trash is graded tight.

Goals use the chain registry names: go_to_fridge, go_to_human, go_to_door,
collect_trash (resolve_goal handles each). Fixed-fixture goals vary only by the random
start; collect_trash also varies the pile per seed.

    python3 scripts/diagnose_reach.py --checkpoint checkpoints/run314_q_model_best.pt \
        --readout softmax_rel --reach 31 --episodes 30
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import homebot  # noqa: F401
from evaluate import load_q_model, process_observation
from chained_eval import run_leg
from goal_geometry import distance, eval_step_budget
from motion import MotionState
from task_chain import resolve_goal


def _min_dist(positions, gx, gy):
    return min(distance(x, y, gx, gy) for (x, y) in positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/run314_q_model_best.pt")
    ap.add_argument("--head-norm", action="store_true")
    ap.add_argument("--goal-layers", type=int, default=2)
    ap.add_argument("--head-layers", type=int, default=4)
    ap.add_argument("--readout", default="softmax_rel")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--reach", type=float, default=31.0,
                    help="reach tolerance px (31 = trash; 79 = fixture default)")
    ap.add_argument("--goals", nargs="+",
                    default=["go_to_fridge", "go_to_human", "go_to_door", "collect_trash"])
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=args.goal_layers, head_layers=args.head_layers,
                         head_norm=args.head_norm, use_motion=True)

    print(f"\ncheckpoint: {args.checkpoint} | readout={args.readout} temp={args.temp} | "
          f"reach<={args.reach:.0f}px | {args.episodes} seeds/goal")
    for name in args.goals:
        rows = []  # (reached, min_dist, steps)
        for seed in range(args.episodes):
            raw, _ = env.reset(seed=seed)
            obs = process_observation(raw)
            try:
                gx, gy = resolve_goal(base, name)
            except Exception as e:
                print(f"  {name}: resolve failed ({e})")
                break
            ms = MotionState(base.action_space.n, getattr(model, "motion_window", 1))
            r = base._robot
            budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
            reached, steps, _, positions = run_leg(
                model, env, base, obs, (gx, gy), budget, device,
                args.readout, args.temp, ms, args.reach)
            rows.append((reached, _min_dist(positions, gx, gy), steps))
        if not rows:
            continue
        nreach = sum(1 for x in rows if x[0])
        fails = [x for x in rows if not x[0]]
        mds = sorted(x[1] for x in fails)
        md_str = (f"fail min-dist min={mds[0]:.0f} med={mds[len(mds)//2]:.0f} max={mds[-1]:.0f}"
                  if fails else "no failures")
        print(f"  {name:<14} reach {nreach}/{len(rows)} = {100*nreach/len(rows):3.0f}%   {md_str}")


if __name__ == "__main__":
    main()
