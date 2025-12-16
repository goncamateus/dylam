from typing import Optional

import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.spaces import Box


class Taxi(TaxiEnv):
    metadata = {
        "render_fps": 30,
        "render_modes": ["human", "rgb_array", "ansi"],
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode=render_mode)
        self.reward_space = Box(-1, 1, shape=(3,))
        self.reward_dim = 3
        num_states = 500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        num_actions = 6
        self.cumulative_reward_info = {
            "reward_Energy": 0,
            "reward_Objective": 0,
            "reward_Illegal_action": 0,
            "Original_reward": 0,
        }
        self.initial_state_distrib = np.zeros(num_states)
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = {
                                "decomposed": np.array([-1, 0, 0]),
                                "original": -1,
                            }
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = {
                                        "decomposed": np.array([0, 0, -1]),
                                        "original": -10,
                                    }
                            elif action == 5:  # dropoff
                                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = {
                                        "decomposed": np.array([0, 1, 0]),
                                        "original": 20,
                                    }
                                elif (taxi_loc in self.locs) and pass_idx == 4:
                                    new_pass_idx = self.locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = {
                                        "decomposed": np.array([0, 0, -1]),
                                        "original": -10,
                                    }
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )

        self.initial_state_distrib /= self.initial_state_distrib.sum()

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.cumulative_reward_info = {
            "reward_Energy": 0,
            "reward_Objective": 0,
            "reward_Illegal_action": 0,
            "Original_reward": 0,
        }
        return result

    def step(self, action):
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        _, state, reward, termination = transitions[i]
        self.s = state
        self.lastaction = action

        if self.render_mode == "human":
            self.render()

        self.cumulative_reward_info["reward_Energy"] += reward["decomposed"][0]
        self.cumulative_reward_info["reward_Objective"] += reward["decomposed"][1]
        self.cumulative_reward_info["reward_Illegal_action"] += reward["decomposed"][2]
        self.cumulative_reward_info["Original_reward"] += reward["original"]
        return (
            int(state),
            reward["decomposed"],
            termination,
            False,
            self.cumulative_reward_info,
        )


class TaxiGrid(Taxi):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.observation_space = Box(0, 5, shape=(4,), dtype=int)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs = np.array(list(self.decode(obs)), dtype=int)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = np.array(list(self.decode(obs)), dtype=int)
        return obs, reward, terminated, truncated, info
