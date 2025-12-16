from typing import List, Union

import mo_gymnasium as mogym
import numpy as np
from pymoo.indicators.hv import HV


def extract_mean_reward_and_info(rewards, informations):
    mean_rewards = np.mean(rewards, axis=0)
    mean_infos = {
        key: np.mean([info[key] for info in informations])
        for key in informations[0].keys()
    }
    return mean_rewards, mean_infos


def episode_eval(env_id, agent, lambdas):
    env = env = mogym.make(env_id)
    observation, _ = env.reset()
    epi_reward = 0
    gamma = 1
    done = False
    while not done:
        observation = observation.reshape(1, -1)
        action = agent.get_action(observation, lambdas)[0]
        next_observation, reward, done, truncation, _ = env.step(action)
        epi_reward += reward * gamma
        gamma *= agent.gamma
        done = done or truncation
        observation = next_observation
    return epi_reward


def eval_policy(env_id, agent, lambdas, num_episodes):
    rewards = [episode_eval(env_id, agent, lambdas) for _ in range(num_episodes)]
    return np.mean(rewards, axis=0)


def get_non_pareto_dominated_inds(
    candidates: Union[np.ndarray, List], remove_duplicates: bool = True
) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(
        candidates, return_index=True, return_inverse=True, return_counts=True, axis=0
    )

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(
    candidates: Union[np.ndarray, List], remove_duplicates: bool = True
) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[
        get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)
    ]


def get_hv(points, ref_point):
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)
