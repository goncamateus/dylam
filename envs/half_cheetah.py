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
            "reward_Final_position": 0,
            "Original_reward": 0,
        }
        self.max_run = 16 * self._forward_reward_weight
        self.min_ctrl = self.action_space.shape[0] * self._ctrl_cost_weight

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_run": 0,
            "reward_ctrl": 0,
            "reward_Final_position": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = np.zeros(2)
        # Forward reward
        reward[0] = info["reward_run"] / self.max_run
        # Control reward
        reward[1] = info["reward_ctrl"] / self.min_ctrl

        self.cumulative_reward_info["reward_run"] += reward[0]
        self.cumulative_reward_info["reward_ctrl"] += reward[1]
        self.cumulative_reward_info["Original_reward"] += (
            reward * np.array([self.max_run, self.min_ctrl])
        ).sum()
        self.cumulative_reward_info["reward_Final_position"] = info["x_position"]
        return observation, reward, terminated, truncated, self.cumulative_reward_info


class HalfCheetahEfficiency(HalfCheetah):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_efficency = 2.55
        self.cumulative_reward_info["reward_efficiency"] = 0

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        self.cumulative_reward_info["reward_efficiency"] = 0
        return res

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        speed = reward[0]
        cost = -reward[1]
        reward_efficiency = speed / cost
        reward_efficiency = reward_efficiency / self.max_efficency
        reward[1] = reward_efficiency
        self.cumulative_reward_info["reward_efficiency"] += reward_efficiency
        return observation, reward, terminated, truncated, self.cumulative_reward_info
