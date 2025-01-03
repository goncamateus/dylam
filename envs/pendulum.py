import numpy as np

from gymnasium.spaces import Box
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class Pendulum(PendulumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(
            low=np.array([-np.pi, -8, -2]),
            high=np.array([0, 0, 0]),
            shape=(3,),
            dtype=float,
        )
        self.reward_dim = 3
        self.cumulative_reward_info = {
            "reward_Theta": 0,
            "reward_Angular_vel": 0,
            "reward_Torque": 0,
            "Original_reward": 0,
        }

    def step(self, action):
        state, reward, termination, truncated, info = super().step(action)
        theta = angle_normalize(np.arccos(state[0]))
        angular_vel = state[2]
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        reward_vec = np.zeros(3, dtype=np.float32)
        reward_vec[0] = abs(theta) / np.pi
        reward_vec[1] = abs(angular_vel) / 8
        reward_vec[2] = abs(torque) / 2
        reward_vec *= -1
        self.cumulative_reward_info["reward_Theta"] += reward_vec[0]
        self.cumulative_reward_info["reward_Angular_vel"] += reward_vec[1]
        self.cumulative_reward_info["reward_Torque"] += reward_vec[2]
        self.cumulative_reward_info["Original_reward"] += reward

        return state, reward_vec, termination, truncated, self.cumulative_reward_info

    def reset(self, *args, **kwargs):
        self.cumulative_reward_info = {
            "reward_Theta": 0,
            "reward_Angular_vel": 0,
            "reward_Torque": 0,
            "Original_reward": 0,
        }
        return super().reset(*args, **kwargs)
