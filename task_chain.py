"""The real multi-task metric: a static chain of go-to subgoals scored out of N.

For now the chain is a fixed list (stand-in for the LLM that will eventually
string subgoals together); score = how many legs the navigator reaches in one
episode, pose persisting leg-to-leg. Top score = len(chain).

Shared by the in-training TB metric (agent.chain_eval) and the offline harness
(chained_eval.py) so both report the identical number.
"""

# get_trash >> go_to_fridge >> go_to_human >> go_to_door >> go_to_human
DEFAULT_CHAIN = ["collect_trash", "go_to_fridge", "go_to_human",
                 "go_to_door", "go_to_human"]


def resolve_goal(base, name):
    """Map a chain subgoal name to pixel (x, y). go_to_human -> the recliner
    (the human's seat); every other name goes through the env goal registry.
    Resolve all legs up front (before stepping) so incidental trash pickup can't
    empty trash_positions mid-chain."""
    if name == "go_to_human":
        col, row = base._map.fixtures["recliner"]
        return tuple(float(v) for v in base._map.tile_to_pixel(col, row))
    return tuple(float(v) for v in base.goal_to_coordinates(name))
