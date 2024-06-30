import numpy as np

from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class HalfCheetah(HalfCheetahEnv, EzPickle):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, True, **kwargs)
        self.reward_dim = 2
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_run": 0,
            "reward_ctrl": 0,
            "Original_reward": 0,
        }

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_run": 0,
            "reward_ctrl": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = np.zeros(2)
        # Forward reward
        reward[0] = info["reward_run"]
        # Control reward
        reward[1] = info["reward_ctrl"]

        self.cumulative_reward_info["reward_run"] += (
            reward[0] / self._forward_reward_weight
        )
        self.cumulative_reward_info["reward_ctrl"] += reward[1] / self._ctrl_cost_weight
        self.cumulative_reward_info["Original_reward"] += (
            reward * np.array([self._forward_reward_weight, self._ctrl_cost_weight])
        ).sum()

        return observation, reward, terminated, truncated, self.cumulative_reward_info
