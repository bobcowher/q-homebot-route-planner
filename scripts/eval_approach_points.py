"""Approach-point hypothesis test.

chained_eval showed wall-pressed/corridor fixtures (fridge, door) fail while the
open-area recliner succeeds. Hypothesis: the goal coordinate is the fixture
CENTROID, which sits inside the fixture footprint / against a wall, so the robot
can't get within GOAL_THRESHOLD of it — even though it can reach the floor right
in front of the fixture. Fix would be to define fixture goals as a reachable
APPROACH POINT (nearest walkable tile), the way a real location registry does.

This drives the navigator to centroid vs approach-point with the SAME start seeds
and reports reach% plus the median closest-approach distance to the centroid, so
we can see whether the robot is physically there but just outside threshold.

    conda run -n sac-homebot python scripts/eval_approach_points.py \
        --checkpoint checkpoints/rg_q_model_best.pt --goal-layers 2 --head-layers 4
"""
import argparse
import os
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch

import homebot  # noqa: F401
from homebot.goals import GOAL_THRESHOLD
from evaluate import load_q_model, process_observation
from goal_geometry import world_coords, distance, eval_step_budget

# fixture name -> goal name (for reporting); we resolve the fixture centroid directly.
FIXTURES = ["fridge", "door", "recliner"]


def nearest_floor(base, cx, cy):
    """Nearest valid floor-tile pixel center to (cx, cy) — the approach point."""
    best, best_d = None, 1e18
    for col, row in base._map.valid_floor_tiles():
        px, py = base._map.tile_to_pixel(col, row)
        d = distance(px, py, cx, cy)
        if d < best_d:
            best, best_d = (float(px), float(py)), d
    return best, best_d


def drive_to(env, base, model, target, device, seed, budget_mult=1.0, start_radius=0.0):
    """One episode: reset(seed), greedily drive toward target. Returns
    (reached_target, min_dist_to_target). If start_radius>0, the robot is
    repositioned to a floor tile within that many px of the target (isolates
    local approach from long cross-room routing)."""
    raw = env.reset(seed=seed)[0]
    obs = process_observation(raw)
    r = base._robot
    if start_radius > 0:
        near = [base._map.tile_to_pixel(c, row) for c, row in base._map.valid_floor_tiles()
                if 79.0 < distance(*base._map.tile_to_pixel(c, row), target[0], target[1]) <= start_radius]
        if near:
            px, py = near[seed % len(near)]
            r.x, r.y = float(px), float(py)
            obs = process_observation(base._get_obs())
    budget = max(1, int(eval_step_budget(distance(r.x, r.y, target[0], target[1])) * budget_mult))
    min_d = distance(r.x, r.y, target[0], target[1])
    for _ in range(budget):
        with torch.no_grad():
            obs_t = obs.unsqueeze(0).float().to(device) / 255.0
            gv = world_coords(r.x, r.y, target[0], target[1])
            gt = torch.as_tensor(gv, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(model(obs_t, gt).argmax(dim=1).item())
        obs = process_observation(env.step(action)[0])
        d = distance(r.x, r.y, target[0], target[1])
        min_d = min(min_d, d)
        if d <= GOAL_THRESHOLD:
            return True, min_d
    return False, min_d


def eval_target(env, base, model, target, device, episodes, seed0, budget_mult, start_radius=0.0):
    reaches, min_ds = 0, []
    for i in range(episodes):
        reached, md = drive_to(env, base, model, target, device, seed0 + i,
                               budget_mult, start_radius)
        reaches += int(reached)
        min_ds.append(md)
    return 100.0 * reaches / episodes, statistics.median(min_ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/rg_q_model_best.pt")
    p.add_argument("--goal-layers", type=int, default=2)
    p.add_argument("--head-layers", type=int, default=4)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-mult", type=float, default=4.0,
                   help="scale per-leg budget; >1 rules out timeout as the cause")
    p.add_argument("--start-radius", type=float, default=150.0,
                   help="near-start spawn radius (isolates local approach from routing)")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
                   obs_resolution=(96, 96), n_trash=2, max_steps=20000,
                   map_name="default", random_start=True)
    base = env.unwrapped
    model = load_q_model(args.checkpoint, env.action_space.n, device,  # type: ignore[union-attr]
                         goal_layers=args.goal_layers, head_layers=args.head_layers)

    print(f"\nreach<= {GOAL_THRESHOLD}px | {args.episodes} ep | same seeds per fixture\n")
    print(f"{'fixture':<10} {'target':<10} {'reach%':>7} {'medMinDist':>11}")
    for name in FIXTURES:
        col, row = base._map.fixtures[name]
        cx, cy = base._map.tile_to_pixel(col, row)
        approach, ad = nearest_floor(base, cx, cy)
        c_reach, c_md = eval_target(env, base, model, (cx, cy), device,
                                    args.episodes, args.seed, args.budget_mult)
        n_reach, n_md = eval_target(env, base, model, (cx, cy), device,
                                    args.episodes, args.seed, args.budget_mult,
                                    start_radius=args.start_radius)
        print(f"{name:<10} {'centroid':<14} {c_reach:>6.0f}% {c_md:>11.0f}")
        print(f"{'':<10} {'near-start':<14} {n_reach:>6.0f}% {n_md:>11.0f}   "
              f"(spawn <= {args.start_radius:.0f}px from centroid)")


if __name__ == "__main__":
    main()
