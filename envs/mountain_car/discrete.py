import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.utils import EzPickle


class MountainCar(MountainCarEnv, EzPickle):
    """
    A multi-objective version of the MountainCar environment, where the goal is to reach the top of the mountain.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) for more information.

    ## Reward space:
    The reward space is a 3D vector containing the time penalty, and penalties for reversing and going forward.
    - time penalty: -1.0 for each time step
    - reverse penalty: -1.0 for each time step the action is 0 (reverse)
    - forward penalty: -1.0 for each time step the action is 2 (forward)
    """

    def __init__(
        self, render_mode: Optional[str] = None, goal_velocity=0, stratified=False
    ):
        super().__init__(render_mode, goal_velocity)
        EzPickle.__init__(self, render_mode, goal_velocity)
        self.stratified = stratified
        self.reward_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([-1, 0, 0]),
            shape=(3,),
            dtype=np.float32,
        )
        self.reward_dim = 3
        self.cumulative_reward_info = {
            "reward_Time": 0,
            "reward_Reverse": 0,
            "reward_Forward": 0,
            "Original_reward": 0,
        }

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_Time": 0,
            "reward_Reverse": 0,
            "reward_Forward": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        # reward = -1.0
        reward = np.zeros(3, dtype=np.float32)
        reward[0] = 0.0 if terminated else -1.0  # time penalty
        reward[1] = 0.0 if action != 0 else -1.0  # reverse penalty
        reward[2] = 0.0 if action != 2 else -1.0  # forward penalty
        self.cumulative_reward_info["reward_Time"] += reward[0]
        self.cumulative_reward_info["reward_Reverse"] += reward[1]
        self.cumulative_reward_info["reward_Forward"] += reward[2]
        self.cumulative_reward_info["Original_reward"] += reward.sum()
        if not self.stratified:
            reward = reward.sum()
        else:
            reward /= np.array([999, 300, 300], dtype=np.float32)

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            False,
            self.cumulative_reward_info,
        )
