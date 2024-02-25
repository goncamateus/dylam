import numpy as np

from gymnasium import Wrapper


class NormalizeReward(Wrapper):
    """
    Normalize the reward to [-1, 1] based on the min and max reward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_max = (0, 0)

    def normalize(self, reward):
        self.min_max = (
            min(self.min_max[0], reward),
            max(self.min_max[1], reward),
        )
        norm = (reward - self.min_max[0]) / (
            self.min_max[1] - self.min_max[0]
        )
        norm = 2 * norm - 1
        return norm

    def step(self, action):
        obs, reward, termination, truncated, info = self.env.step(action)
        return obs, self.normalize(reward), termination, truncated, info


class NormalizeMOReward(Wrapper):
    """
    Normalize the reward to [-1, 1] based on the min and max reward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_dim = env.unwrapped.reward_space.shape[0]
        self.min_max = np.zeros((self.reward_dim, 2))

    def normalize(self, reward):
        self.min_max[:, 0] = np.minimum(self.min_max[:, 0], reward)
        self.min_max[:, 1] = np.maximum(self.min_max[:, 1], reward)

        norm = (reward - self.min_max[:, 0]) / (
            self.min_max[:, 1] - self.min_max[:, 0]
        )
        norm = 2 * norm - 1
        return norm

    def step(self, action):
        obs, reward, termination, truncated, info = self.env.step(action)
        return obs, self.normalize(reward), termination, truncated, info
