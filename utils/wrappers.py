from copy import deepcopy
import time
import gymnasium as gym
import numpy as np

from gymnasium import Wrapper
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.utils import RecordConstructorArgs


class MORecordEpisodeStatistics(RecordEpisodeStatistics, RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward (array)>",
        ...         "dr": "<discounted reward (array)>",
        ...         "l": "<episode length (scalar)>", # contrary to Gymnasium, these are not a numpy array
        ...         "t": "<elapsed time since beginning of episode (scalar)>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of (be careful to first wrap the env into vector before applying MORewordStatistics)::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward (2d array, shape (num_envs, dim_reward))>",
        ...         "dr": "<array of discounted reward (2d array, shape (num_envs, dim_reward))>",
        ...         "l": "<array of episode length (array)>",
        ...         "t": "<array of elapsed time since beginning of episode (array)>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }
    """

    def __init__(self, env: gym.Env, gamma: float = 1.0, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            gamma (float): Discounting factor
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, gamma=gamma, deque_size=deque_size
        )
        RecordEpisodeStatistics.__init__(self, env, deque_size=deque_size)
        # CHANGE: Here we just override the standard implementation to extend to MO
        # We also take care of the case where the env is vectorized
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]
        if self.is_vector_env:
            self.rewards_shape = (self.num_envs, self.reward_dim)
        else:
            self.rewards_shape = (self.reward_dim,)
        self.gamma = gamma

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        # This is very close the code from the RecordEpisodeStatistics wrapper from gym.
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1

        # CHANGE: The discounted returns are also computed here
        self.disc_episode_returns += rewards * np.repeat(
            self.gamma**self.episode_lengths, self.reward_dim
        ).reshape(self.episode_returns.shape)

        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                disc_episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                if self.is_vector_env:
                    for i in range(self.num_envs):
                        if dones[i]:
                            # CHANGE: Makes a deepcopy to avoid subsequent mutations
                            episode_return[i] = deepcopy(self.episode_returns[i])
                            disc_episode_return[i] = deepcopy(
                                self.disc_episode_returns[i]
                            )
                else:
                    episode_return = deepcopy(self.episode_returns)
                    disc_episode_return = deepcopy(self.disc_episode_returns)

                length_eps = np.where(dones, self.episode_lengths, 0)
                time_eps = np.where(
                    dones,
                    np.round(time.perf_counter() - self.episode_start_times, 6),
                    0.0,
                )

                infos["episode"] = {
                    "r": episode_return,
                    "dr": disc_episode_return,
                    "l": length_eps[0] if not self.is_vector_env else length_eps,
                    "t": time_eps[0] if not self.is_vector_env else time_eps,
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = np.zeros(self.reward_dim, dtype=np.float32)
            self.disc_episode_returns[dones] = np.zeros(
                self.reward_dim, dtype=np.float32
            )
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


class NormalizeReward(Wrapper):
    """
    Normalize the reward to [-1, 1] based on the min and max reward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_max = (0, 0)
        self.reward_sum = 0

    def normalize(self, reward):
        self.min_max = (
            min(self.min_max[0], reward),
            max(self.min_max[1], reward),
        )
        norm = (reward - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
        norm = 2 * norm - 1
        return norm

    def step(self, action):
        obs, reward, termination, truncated, info = self.env.step(action)
        normed_reward = self.normalize(reward)
        self.reward_sum += reward
        if termination:
            info.update({f"reward_normed": self.reward_sum})
            self.reward_sum = 0
        return obs, normed_reward, termination, truncated, info


class NormalizeMOReward(Wrapper):
    """
    Normalize the reward to [-1, 1] based on the min and max reward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_dim = env.unwrapped.reward_space.shape[0]
        self.min_max = np.zeros((self.reward_dim, 2))
        self.reward_sum = np.zeros(self.reward_dim)

    def normalize(self, reward):
        self.min_max[:, 0] = np.minimum(self.min_max[:, 0], reward)
        self.min_max[:, 1] = np.maximum(self.min_max[:, 1], reward)

        norm = (reward - self.min_max[:, 0]) / (
            self.min_max[:, 1] - self.min_max[:, 0] + 1e-6
        )
        norm = 2 * norm - 1
        return norm

    def step(self, action):
        obs, reward, termination, truncated, info = self.env.step(action)
        normed_reward = self.normalize(reward)
        self.reward_sum += reward
        if termination:
            for i in range(self.reward_dim):
                info.update({f"reward_{i}_normed": self.reward_sum[i]})
            self.reward_sum = np.zeros(self.reward_dim)
        return obs, normed_reward, termination, truncated, info
