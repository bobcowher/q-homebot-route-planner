import gymnasium as gym

class CardinalActionWrapper(gym.ActionWrapper):
    """Wraps a HomeBot2D environment to restrict the action space to the 4 cardinal directions:
    0 -> 0 (North)
    1 -> 2 (East)
    2 -> 4 (South)
    3 -> 6 (West)
    
    This reduces the action space from 8 to 4, simplifying the Q-value learning and
    preventing diagonal wall-sliding oscillations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self._map = [0, 2, 4, 6]

    def action(self, action):
        return self._map[int(action)]

    def reverse_action(self, action):
        # Maps the 8-directional action back to 4-directional if possible
        try:
            return self._map.index(int(action))
        except ValueError:
            # If it's a diagonal, map to the closest cardinal
            # 1 (NE) -> 0 (N) or 2 (E), etc.
            if action == 1: return 0
            if action == 3: return 1
            if action == 5: return 2
            if action == 7: return 3
            return 0
