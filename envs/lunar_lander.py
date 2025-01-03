import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
)


class LunarLanderStrat(
    LunarLander
):  # no need for EzPickle, it's already in LunarLander
    """
    ## Description
    Multi-objective version of the LunarLander environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/box2d/lunar_lander/) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: -100 if crash, +100 if lands successfully
    - 1: Shaping reward
    - 2: Fuel cost (main engine)
    - 3: Fuel cost (side engine)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_reward_info = {
            "reward_Shaping": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        self.reward_space = gym.spaces.Box(
            low=np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            shape=(4,),
        )
        self.reward_dim = 4
        self.prev_rew = None

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_Shaping": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                s_power = 1.0

        state, reward, termination, truncated, info = super().step(action)
        reward_vec = np.zeros(self.reward_dim)
        # Distance to center
        shaping = -np.sqrt(state[0] * state[0] + state[1] * state[1])*1000
        # Speed discount
        shaping += -np.sqrt(state[2] * state[2] + state[3] * state[3])*1000
        # Angle discount
        shaping += -abs(state[4])*1000
        # Ground Contacts
        shaping += state[6]
        shaping += state[7]
        shaping /= 3
        if self.prev_rew is not None:
            reward_vec[0] = shaping - self.prev_rew

        # Power discount
        reward_vec[1] = -m_power
        reward_vec[2] = -s_power

        # Win/Lost
        if termination:
            self.prev_rew = None
            shaping = 0
            reward_vec = np.zeros(self.reward_dim)
            if self.game_over or abs(state[0]) >= 1.0:
                reward_vec[3] = -1
            if not self.lander.awake:
                reward_vec[3] = 1

        if reward == 0:
            reward_vec = np.zeros(self.reward_dim)

        self.prev_rew = shaping

        self.cumulative_reward_info["reward_Shaping"] += reward_vec[0]
        self.cumulative_reward_info["reward_Power_linear"] += reward_vec[1]
        self.cumulative_reward_info["reward_Power_angular"] += reward_vec[2]
        self.cumulative_reward_info["reward_Goal"] += reward_vec[3]

        self.cumulative_reward_info["Original_reward"] += reward
        info.update(self.cumulative_reward_info)
        return state, reward_vec, termination, truncated, info
