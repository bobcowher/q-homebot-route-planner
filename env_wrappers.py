import gymnasium as gym

class CardinalActionWrapper(gym.ActionWrapper):
    """Wraps a HomeBot2D environment to restrict the action space to the 4 cardinal directions:
    0 -> 0 (North)
    1 -> 2 (East)
    2 -> 4 (South)
    3 -> 6 (West)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self._map = [0, 2, 4, 6]

    def action(self, action):
        return self._map[int(action)]

    def reverse_action(self, action):
        try:
            return self._map.index(int(action))
        except ValueError:
            if action == 1: return 0
            if action == 3: return 1
            if action == 5: return 2
            if action == 7: return 3
            return 0


class FrameSkipWrapper(gym.Wrapper):
    """Wraps a HomeBot2D environment to repeat the chosen action for `skip` steps.
    Accumulates reward over the skipped steps and returns the final observation.
    """
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
