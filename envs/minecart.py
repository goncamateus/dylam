from mo_gymnasium.envs.minecart.minecart import Minecart


class MinecartEnv(Minecart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cumulative_info = {
            "reward_First_minerium": 0,
            "reward_Second_minerium": 0,
            "reward_Fuel": 0,
            "reward_Original": 0,
        }

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.cumulative_info = {
            "reward_First_minerium": 0,
            "reward_Second_minerium": 0,
            "reward_Fuel": 0,
            "reward_Original": 0,
        }
        return obs, info

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        self.cumulative_info["reward_First_minerium"] += reward[0]
        self.cumulative_info["reward_Second_minerium"] += reward[1]
        self.cumulative_info["reward_Fuel"] += reward[2]
        self.cumulative_info["reward_Original"] += reward.sum()
        return obs, reward, termination, truncation, info
