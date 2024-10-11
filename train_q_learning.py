# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import numpy as np

from methods.q_learning import QLearning, DRQ, QDyLam
from utils.experiment import get_experiment, q_make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import QLogger


def get_agent_type(args):
    if args.stratified:
        if args.dylam:
            return QDyLam
        return DRQ
    return QLearning


def train(args, exp_name, logger: QLogger):
    env = q_make_env(args, exp_name)
    agent_type = get_agent_type(args)
    agent = agent_type(args, env.observation_space, env.action_space)

    for episode in range(1, args.total_episodes + 1):
        obs, _ = env.reset()
        termination = False
        truncation = False
        while not (termination or truncation):
            action = agent.get_action(obs)

            next_obs, reward, termination, truncation, info = env.step(action)
            logger.log_episode(info, termination or truncation)

            if args.dylam:
                agent.add_episode_reward(reward, termination, truncation)

            update_values = agent.update(obs, action, reward, next_obs)
            loss_dict = {}
            if args.stratified:
                for i in range(args.num_rewards):
                    loss_dict[f"qf_update_{i}"] = update_values[i]
            else:
                loss_dict["qf_update"] = update_values
            logger.log_losses(loss_dict)
            obs = next_obs

        if args.dylam:
            agent.update_lambdas()
            logger.log_lambdas(agent.lambdas)

        logger.push(episode)
        if episode % 10 == 0:
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
