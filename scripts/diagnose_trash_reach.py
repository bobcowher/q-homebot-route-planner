"""Why does collect_trash fail one pile but not the other? For each scene, drive the
navigator to EACH trash pile independently (reset before each, so pile B's outcome
isn't contaminated by getting stuck at pile A) and record reached + the MIN distance
to the pile achieved. The min-distance distribution distinguishes the failure mode:

  - failed but min_dist ~31-55px  -> gets there, can't CLOSE (terminal precision)
  - failed and min_dist large      -> never approaches (wall-stick / needs a detour
                                       around an obstacle the greedy-to-coord policy
                                       can't take)

Every trash tile is floor and the robot fits on it (radius 15 < 16px half-tile), so
physical reachability is not the question -- this isolates navigation vs precision.

    python3 scripts/diagnose_trash_reach.py --checkpoint checkpoints/run324_macro_q_model_best.pt \
        --head-norm --readout greedy --episodes 20
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


def _min_dist(positions, gx, gy):
    return min(distance(x, y, gx, gy) for (x, y) in positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/run324_macro_q_model_best.pt")
    ap.add_argument("--head-norm", action="store_true")
    ap.add_argument("--goal-layers", type=int, default=2)
    ap.add_argument("--head-layers", type=int, default=4)
    ap.add_argument("--readout", default="greedy")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,
                         goal_layers=args.goal_layers, head_layers=args.head_layers,
                         head_norm=args.head_norm, use_motion=True)

    rows = []  # (seed, pile_idx, reached, min_dist, steps)
    for seed in range(args.episodes):
        # how many piles this scene has (deterministic for the seed)
        env.reset(seed=seed)
        n_piles = len(base._task_manager.trash_positions)
        for i in range(n_piles):
            raw, _ = env.reset(seed=seed)          # same scene, fresh start each pile
            obs = process_observation(raw)
            piles = [base._map.tile_to_pixel(*p) for p in base._task_manager.trash_positions]
            gx, gy = float(piles[i][0]), float(piles[i][1])
            ms = MotionState(base.action_space.n, getattr(model, "motion_window", 1))
            r = base._robot
            budget = max(1, int(eval_step_budget(distance(r.x, r.y, gx, gy))))
            reached, steps, _, positions = run_leg(
                model, env, base, obs, (gx, gy), budget, device,
                args.readout, args.temp, ms, TRASH_REACH)
            rows.append((seed, i, reached, _min_dist(positions, gx, gy), steps))

    reached = [r for r in rows if r[2]]
    failed = [r for r in rows if not r[2]]
    print(f"\ncheckpoint: {args.checkpoint} | readout={args.readout} | "
          f"reach<={TRASH_REACH}px | {len(rows)} pile-drives")
    print(f"  reached: {len(reached)}/{len(rows)} = {100*len(reached)/len(rows):.0f}%")
    if failed:
        mds = sorted(r[3] for r in failed)
        close = sum(1 for d in mds if d <= 55)
        print(f"  failed:  {len(failed)}  | min-dist to pile: "
              f"min={mds[0]:.0f} median={mds[len(mds)//2]:.0f} max={mds[-1]:.0f}")
        print(f"    of failures, got within 55px (close, can't close): "
              f"{close}/{len(failed)}  | stuck far (>55px): {len(failed)-close}/{len(failed)}")
    # per-pile-index reach (is one slot systematically worse?)
    for i in sorted(set(r[1] for r in rows)):
        sub = [r for r in rows if r[1] == i]
        sr = sum(1 for r in sub if r[2])
        print(f"  pile #{i}: reached {sr}/{len(sub)} = {100*sr/len(sub):.0f}%")


if __name__ == "__main__":
    main()
