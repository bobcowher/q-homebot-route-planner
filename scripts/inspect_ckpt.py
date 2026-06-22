"""Inspect saved q_model checkpoints: print every 2D (linear) weight shape and
any scalar config entries. Compare a known window=1 checkpoint (run314) against
a known window=8 one (run318) to find which linear layer's in_features encodes
the motion width (window=1 -> 10 dims, window=8 -> 12 dims), then read run320."""
import sys
import torch


def sd_of(obj):
    if isinstance(obj, dict):
        for key in ("q_model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], {k: v for k, v in obj.items() if k != key}
        return obj, {}
    return obj.state_dict(), {}


for path in sys.argv[1:]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd, extra = sd_of(obj)
    print(f"\n=== {path} ===")
    scalars = {k: v for k, v in sd.items()
               if torch.is_tensor(v) and v.numel() <= 8 and v.dim() <= 1}
    for k, v in scalars.items():
        print(f"  scalar {k}: {v.tolist()}")
    for k, v in extra.items():
        if not torch.is_tensor(v):
            print(f"  extra {k}: {v}")
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dim() == 2:
            print(f"  linear {k}: in={v.shape[1]} out={v.shape[0]}")
