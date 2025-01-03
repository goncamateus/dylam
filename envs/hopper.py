import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class Hopper(HopperEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HopperEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, True, **kwargs)
        self.cost_objetive = True
        self.reward_dim = 4
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Forward": 0,
            "reward_Jump": 0,
            "reward_Energy": 0,
            "reward_Healthy": 0,
            "Original_reward": 0,
        }

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_Forward": 0,
            "reward_Jump": 0,
            "reward_Energy": 0,
            "reward_Healthy": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        healthy_reward = self.healthy_reward

        observation = self._get_obs()

        terminated = self.terminated

        z = self.data.qpos[1]
        height = z - self.init_qpos[1]
        energy_cost = np.sum(np.square(action))

        vec_reward = np.array(
            [
                x_velocity / 2.5,
                height * 4,
                -energy_cost / 3,
                healthy_reward,
            ],
            dtype=np.float32,
        )

        self.cumulative_reward_info["reward_Forward"] += x_velocity / 2.5
        self.cumulative_reward_info["reward_Jump"] += height * 4
        self.cumulative_reward_info["reward_Energy"] += -energy_cost / 3
        self.cumulative_reward_info["reward_Healthy"] += healthy_reward
        self.cumulative_reward_info["Original_reward"] += (
            x_velocity + (10 * height) - energy_cost + healthy_reward
        )

        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, terminated, False, self.cumulative_reward_info
