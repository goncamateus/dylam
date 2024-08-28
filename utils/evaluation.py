import mo_gymnasium as mogym
import numpy as np


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
