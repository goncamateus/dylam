import mo_gymnasium as mogym
import numpy as np


def extract_mean_reward_and_info(rewards, informations):
    mean_rewards = np.mean(rewards, axis=0)
    mean_infos = {
        key: np.mean([info[key] for info in informations])
        for key in informations[0].keys()
    }
    return mean_rewards, mean_infos


def eval_policy(env_id, agent, lambdas, num_episodes):
    env = env = mogym.make(env_id)
    env = mogym.MORecordEpisodeStatistics(env)
    rewards = []
    informations = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        informations.append(info)
        done = False
        epi_reward = 0
        while not done:
            observation = observation.reshape(1, -1)
            action = agent.get_action(observation, lambdas)[0]
            next_observation, reward, done, truncation, info = env.step(action)
            epi_reward += reward
            done = done or truncation
            observation = next_observation
            if done:
                for key, value in info.items():
                    if "reward" in key:
                        informations[-1][key] = value
        rewards.append(epi_reward)
    return extract_mean_reward_and_info(rewards, informations)
