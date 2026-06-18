"""Quantify how well uniform random goal/spawn placement covers the fixtures.

The chain fails on go_to_fridge and go_to_door (both 0%) while open/central
goals work. Hypothesis: uniform tile sampling rarely produces a goal NEAR the
fridge/door (corner alcove + narrow doorway), so those approaches are starved
in training. This counts, per fixture, how many valid floor tiles fall within
GOAL_THRESHOLD — i.e. how often a uniform goal teaches that approach.

    conda run -n sac-homebot python scripts/check_placement.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import gymnasium as gym
import homebot  # noqa: F401
from homebot.goals import GOAL_THRESHOLD

env = gym.make("HomeBot2D-V1", render_mode="rgb_array", action_mode="discrete",
               obs_resolution=(96, 96), n_trash=2, map_name="default",
               random_start=True)
base = env.unwrapped
m = base._map

tiles = m.valid_floor_tiles()
n = len(tiles)
pix = [m.tile_to_pixel(c, r) for (c, r) in tiles]
print(f"valid floor tiles: {n}  (tile_size={m.tile_size}, threshold={GOAL_THRESHOLD}px)")

# Map bounds for region split (kitchen = north/east, living = south/west).
cols = [c for c, r in tiles]
rows = [r for c, r in tiles]
print(f"col range {min(cols)}..{max(cols)}  row range {min(rows)}..{max(rows)}")

print("\nper-fixture goal coverage (tiles within GOAL_THRESHOLD of fixture centre):")
for name in ("fridge", "recliner", "door"):
    if name not in m.fixtures:
        continue
    fx, fy = m.tile_to_pixel(*m.fixtures[name])
    near = [1 for (px, py) in pix
            if math.sqrt((px - fx) ** 2 + (py - fy) ** 2) <= GOAL_THRESHOLD]
    k = len(near)
    print(f"  {name:<9} tile={m.fixtures[name]}  reachable goal-tiles={k:>3}/{n}  "
          f"= {100*k/n:4.1f}% of uniform goals")

# How many tiles sit in each room band, to see the spatial skew.
print("\nrow histogram (tiles per row — shows where the floor mass is):")
for rr in range(min(rows), max(rows) + 1):
    cnt = sum(1 for r in rows if r == rr)
    print(f"  row {rr:>2}: {'#' * cnt} {cnt}")
