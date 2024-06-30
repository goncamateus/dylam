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
            "reward_Range/run": 0,
            "reward_Range/ctrl": 0,
            "Original_reward": 0,
        }

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_run": 0,
            "reward_ctrl": 0,
            "reward_Range/run": 0,
            "reward_Range/ctrl": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = np.zeros(2)
        # Forward reward
        max_run = 16 * self._forward_reward_weight
        reward[0] = info["reward_run"] / max_run
        # Control reward
        min_ctrl = 6 * self._ctrl_cost_weight
        reward[1] = info["reward_ctrl"] / min_ctrl

        self.cumulative_reward_info["reward_run"] += reward[0]
        self.cumulative_reward_info["reward_ctrl"] += reward[1]
        self.cumulative_reward_info["reward_Range/run"] = max(
            self.cumulative_reward_info["reward_Range/run"],
            reward[0],
        )
        self.cumulative_reward_info["reward_Range/ctrl"] = min(
            self.cumulative_reward_info["reward_Range/ctrl"],
            reward[1],
        )
        self.cumulative_reward_info["Original_reward"] += (
            reward * np.array([max_run, min_ctrl])
        ).sum()
        reward = reward * np.array([max_run, min_ctrl])
        return observation, reward, terminated, truncated, self.cumulative_reward_info
