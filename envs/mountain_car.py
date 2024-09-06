from mo_gymnasium.envs.mountain_car.mountain_car import MOMountainCar


class MountainCar(MOMountainCar):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_reward_info = {
            "reward_time": 0,
            "reward_reverse": 0,
            "reward_forward": 0,
            "Original_reward": 0,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = super().reset(seed=seed, options=options)
        self.cumulative_reward_info = {
            "reward_time": 0,
            "reward_reverse": 0,
            "reward_forward": 0,
            "Original_reward": 0,
        }
        return result

    def step(self, action: int):
        state, reward, terminated, truncated, info = super().step(action)
        self.cumulative_reward_info["reward_time"] += reward[0]
        self.cumulative_reward_info["reward_reverse"] += reward[1]
        self.cumulative_reward_info["reward_forward"] += reward[2]
        self.cumulative_reward_info["Original_reward"] += -1
        return state, reward, terminated, truncated, self.cumulative_reward_info
