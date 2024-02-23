from mo_gymnasium.envs.mujoco.reacher import MOReacherEnv


class Reacher(MOReacherEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_reward_info = {
            "Original_reward": 0,
            "reward_Arm1": 0,
            "reward_Arm2": 0,
            "reward_Arm3": 0,
            "reward_Arm4": 0,
        }

    def reset(self, *args, **kwargs):
        obs, _ = super().reset(*args, **kwargs)
        self.cumulative_reward_info = {
            "Original_reward": 0,
            "reward_Arm1": 0,
            "reward_Arm2": 0,
            "reward_Arm3": 0,
            "reward_Arm4": 0,
        }
        return obs, self.cumulative_reward_info

    def step(self, action):
        state, reward, termination, truncated, _ = super().step(action)
        for i in range(4):
            self.cumulative_reward_info[f"reward_Arm{i+1}"] = reward[i]
        self.cumulative_reward_info["Original_reward"] = reward.sum()
        return state, reward, termination, truncated, self.cumulative_reward_info
