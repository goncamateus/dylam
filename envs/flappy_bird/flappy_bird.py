import gymnasium as gym
import numpy as np


from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv


class FlappyBird(FlappyBirdEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_dim = 4
        self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Pipe": 0,
            "reward_Healthy": 0,
            "reward_Top": 0,
            "reward_Dead": 0,
            "Original_reward": 0,
        }

    def reset(self, *args, **kwargs):
        self.cumulative_reward_info = {
            "reward_Pipe": 0,
            "reward_Healthy": 0,
            "reward_Top": 0,
            "reward_Dead": 0,
            "Original_reward": 0,
        }
        return super().reset(*args, **kwargs)

    def step(self, action):
        observation, reward, termination, truncated, info = super().step(action)
        reward_vec = np.zeros(self.reward_dim)
        if reward == 1:
            reward_vec[0] = 1
        elif reward == 0.1:
            reward_vec[1] = 1
        elif reward == -0.5:
            reward_vec[2] = -1
        elif reward == -1:
            reward_vec[3] = -1
        self.cumulative_reward_info["reward_Pipe"] += reward_vec[0]
        self.cumulative_reward_info["reward_Healthy"] += reward_vec[1]
        self.cumulative_reward_info["reward_Top"] += reward_vec[2]
        self.cumulative_reward_info["reward_Dead"] += reward_vec[3]
        self.cumulative_reward_info["Original_reward"] += reward
        return (
            observation,
            reward_vec,
            termination,
            truncated,
            self.cumulative_reward_info,
        )
