"""A* path planning over the known tilemap -> short pixel waypoints for the reactive
navigator. The architectural fix for the navigator's long-tail unreliability
([[navigator-architecture-limit]]): the navigator is reliable on SHORT hops but
stalls on far/arbitrary goals (its greedy-over-learned-Q field has curl). So plan the
global route on the map (what a real robot does with SLAM) and hand the navigator a
sequence of nearby waypoints, each in its reliable regime.

Walkable = the same floor set trash spawns on / the robot can occupy:
(tiles == FLOOR) & (~solid). 8-connected A* with a Euclidean heuristic.
"""
import heapq
import math

FLOOR = 0


def _walkable(base_map):
    return (base_map.tiles == FLOOR) & (~base_map.solid)  # bool [rows, cols]


def _nearest_walkable(walk, r, c):
    """Snap (r,c) to the nearest walkable tile (BFS ring) -- robot/goal pixels can
    land a hair into a non-walkable tile."""
    rows, cols = walk.shape
    if 0 <= r < rows and 0 <= c < cols and walk[r, c]:
        return (r, c)
    for rad in range(1, max(rows, cols)):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and walk[rr, cc]:
                    return (rr, cc)
    return (r, c)


def plan_tiles(base_map, start_px, goal_px):
    """A* on the tile grid; returns a list of (col,row) tiles start..goal, or [] if
    no path."""
    walk = _walkable(base_map)
    ts = base_map.tile_size
    s = _nearest_walkable(walk, int(start_px[1]) // ts, int(start_px[0]) // ts)
    g = _nearest_walkable(walk, int(goal_px[1]) // ts, int(goal_px[0]) // ts)
    if s == g:
        return [s]
    rows, cols = walk.shape
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_h = [(h(s, g), 0.0, s)]
    came, gscore = {}, {s: 0.0}
    while open_h:
        _, gc, cur = heapq.heappop(open_h)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return [(c, r) for (r, c) in path]  # (row,col) -> (col,row)
        if gc > gscore.get(cur, float("inf")):
            continue
        for dr, dc, cost in nbrs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols and walk[nr, nc]):
                continue
            if dr != 0 and dc != 0:  # no diagonal cutting through a wall corner
                if not (walk[cur[0] + dr, cur[1]] and walk[cur[0], cur[1] + dc]):
                    continue
            ng = gc + cost
            nxt = (nr, nc)
            if ng < gscore.get(nxt, float("inf")):
                gscore[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_h, (ng + h(nxt, g), ng, nxt))
    return []


def plan_waypoints(base_map, start_px, goal_px, stride=2):
    """Pixel waypoints from start toward goal, every `stride` tiles (short hops the
    navigator can do reliably), with the EXACT goal pixel as the final waypoint.
    Returns [] if no path (caller falls back to driving the goal directly)."""
    tiles = plan_tiles(base_map, start_px, goal_px)
    if len(tiles) <= 1:
        return []
    pts = [base_map.tile_to_pixel(c, r) for (c, r) in tiles[1::stride]]
    pts = [(float(x), float(y)) for (x, y) in pts]
    goal = (float(goal_px[0]), float(goal_px[1]))
    if not pts or pts[-1] != goal:
        pts.append(goal)
    return pts
