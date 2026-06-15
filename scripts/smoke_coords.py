"""Smoke-test the coordinate-reframing QModel (goal_dim=4) at both depths.

    conda run -n sac-homebot python scripts/smoke_coords.py

Checks: shallow (1/1) and deep (2/2) build, forward to the right shape, and that
the goal_scale buffer matches goal_dim. Catches shape/wiring bugs before a run.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.q_model import QModel

obs = torch.zeros(2, 3, 96, 96)
goal = torch.tensor([[100.0, 50.0, 700.0, 400.0],
                     [0.0, 0.0, 864.0, 576.0]], dtype=torch.float32)

for name, gl, hl in [("shallow", 1, 1), ("deep", 2, 2)]:
    m = QModel(action_dim=8, goal_dim=4, goal_layers=gl, head_layers=hl)
    out = m(obs, goal)
    assert out.shape == (2, 8), f"{name}: bad output shape {out.shape}"
    assert m.goal_scale.shape[0] == 4, f"{name}: goal_scale width {m.goal_scale.shape}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: out={tuple(out.shape)} params={n_params:,} OK")

print("COORD SMOKE OK")
