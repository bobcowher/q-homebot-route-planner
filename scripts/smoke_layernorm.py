"""Smoke-test the LayerNorm head knob on the coord rep.

    conda run -n sac-homebot python scripts/smoke_layernorm.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.q_model import QModel

obs = torch.zeros(2, 3, 96, 96)
goal = torch.tensor([[100.0, 50.0, 700.0, 400.0],
                     [0.0, 0.0, 864.0, 576.0]], dtype=torch.float32)

# off: no head_norms
m0 = QModel(action_dim=8, goal_dim=4, goal_layers=2, head_layers=4, head_norm=False)
assert m0.head_norms is None, "head_norms should be None when off"
assert m0(obs, goal).shape == (2, 8)

# on: one LayerNorm per head layer
m1 = QModel(action_dim=8, goal_dim=4, goal_layers=2, head_layers=4, head_norm=True)
assert m1.head_norms is not None and len(m1.head_norms) == 4, "expected 4 head LayerNorms"
assert all(isinstance(ln, nn.LayerNorm) for ln in m1.head_norms)
out = m1(obs, goal)
assert out.shape == (2, 8), f"bad output shape {out.shape}"
n_params = sum(p.numel() for p in m1.parameters())
print(f"head_norm on: {len(m1.head_norms)} LayerNorms, out={tuple(out.shape)}, params={n_params:,} OK")
print("LAYERNORM SMOKE OK")
