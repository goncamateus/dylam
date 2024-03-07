import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv, mass_center
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class Humanoid(HumanoidEnv, EzPickle):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, True, **kwargs)
        self.cost_objetive = True
        self.reward_dim = 3
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Forward": 0,
            "reward_Energy": 0,
            "reward_Healthy": 0,
            "Original_reward": 0,
        }

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_Forward": 0,
            "reward_Energy": 0,
            "reward_Healthy": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, _ = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        observation = self._get_obs()

        terminated = self.terminated
        ori_energy_cost = self.control_cost(action)
        energy_cost = ori_energy_cost / self._ctrl_cost_weight

        vec_reward = np.array(
            [
                forward_reward,
                -energy_cost / 2.72,
                healthy_reward / 5,
            ],
            dtype=np.float32,
        )

        self.cumulative_reward_info["reward_Forward"] += forward_reward
        self.cumulative_reward_info["reward_Energy"] += -energy_cost
        self.cumulative_reward_info["reward_Healthy"] += healthy_reward
        self.cumulative_reward_info["Original_reward"] += (
            (self._forward_reward_weight * forward_reward)
            + self.healthy_reward
            - ori_energy_cost
        )

        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, terminated, False, self.cumulative_reward_info
