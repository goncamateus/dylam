# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np

from methods.dqn import DQN, DQNStrat
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import DQNLogger


def train(args, exp_name, logger: DQNLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )
    if args.stratified:
        agent = DQNStrat(
            args,
            envs.single_observation_space,
            envs.single_action_space,
        )
    else:
        agent = DQN(args, envs.single_observation_space, envs.single_action_space)

    obs, _ = envs.reset()
    for global_step in range(1, args.total_timesteps + 1):
        actions = agent.get_action(obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)
        obs = next_obs

        if args.dylam:
            agent.add_episode_rewards(rewards, terminations, truncations)
            agent.update_lambdas()
        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and global_step % args.update_frequency == 0
        ):
            losses = agent.update(args.batch_size)

            if global_step % args.target_network_frequency == 0:
                if args.stratified:
                    for i in range(args.num_rewards):
                        agent.target_q_networks[i].sync(args.tau)
                else:
                    agent.target_q_network.sync(args.tau)

            if global_step % 100 == 0:
                loss_dict = {}
                if args.stratified:
                    for i in range(args.num_rewards):
                        loss_dict[f"qf_loss_{i}"] = losses[i]
                else:
                    loss_dict["qf_loss"] = losses
                logger.log_losses(loss_dict)
                if args.dylam:
                    logger.log_lambdas(agent.lambdas)

        logger.push(global_step)
        if global_step % 9999 == 0:
            agent.save(f"models/{exp_name}/")

    logger.log_artifact()
    envs.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = DQNLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
