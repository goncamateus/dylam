# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import numpy as np

from methods.q_learning import QLearning, drQ, QDyLam
from utils.experiment import get_experiment, q_make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import QLogger


def get_agent_type(args):
    if args.stratified:
        if args.dylam:
            return QDyLam
        return drQ
    return QLearning


def train(args, exp_name, logger: QLogger):
    env = q_make_env(args, 0, exp_name)()
    agent_type = get_agent_type(args)
    agent = agent_type(args, env.observation_space, env.action_space)

    for episode in range(args.total_episodes):
        obs, _ = env.reset()
        termination = False
        truncation = False
        while not (termination or truncation):
            action = agent.get_action(obs)

            next_obs, reward, termination, truncation, info = env.step(action)
            logger.log_episode(info, termination or truncation)

            if args.dylam:
                agent.add_episode_reward(reward, termination, truncation)

            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs

        if args.dylam:
            agent.update_lambdas()
            logger.log_lambdas(agent.lambdas)

        logger.push(episode)
        if episode % 9 == 0:
            agent.save(f"models/{exp_name}/")

    logger.log_artifact()
    env.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = QLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
