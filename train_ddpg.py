# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import wandb

import envs

from methods.ddpg import DDPG, DDPGStrat
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import DDPGLogger
from utils.ou_noise import OUNoise


def train(args, exp_name, logger: DDPGLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )
    if args.stratified:
        agent = DDPGStrat(
            args,
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dim=args.hidden_dim,
        )
    else:
        agent = DDPG(args, envs.single_observation_space, envs.single_action_space)
    ou_noise = [
        OUNoise(envs.single_action_space.shape[0], sigma=0.8)
        for _ in range(args.num_envs)
    ]
    obs, _ = envs.reset()
    for noise in ou_noise:
        noise.reset()
    for global_step in range(args.total_timesteps):
        actions = np.zeros((args.num_envs, envs.single_action_space.shape[0]))
        if global_step > args.learning_starts:
            actions = agent.get_action(obs)
        for i, noise in enumerate(ou_noise):
            actions[i] = np.clip(actions[i] + noise.sample(), -1, 1)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards)
        for i in range(args.num_envs):
            if terminations[i] or truncations[i]:
                ou_noise[i].reset()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)
        obs = next_obs

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and global_step % args.update_frequency == 0
        ):
            if args.dylam:
                agent.add_episode_rewards(rewards, terminations, truncations)
                agent.update_lambdas()
            update_actor = global_step % args.policy_frequency == 0
            losses = agent.update(args.batch_size, update_actor)

            if global_step % args.target_network_frequency == 0:
                agent.actor_target.sync(args.tau)
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                loss_dict = {
                    "policy_loss": losses[0],
                    "qf_loss": losses[1],
                }
                logger.log_losses(loss_dict)
                if args.dylam:
                    logger.log_lambdas(agent.lambdas)
            if global_step % args.sigma_decay == 0:
                for i, noise in enumerate(ou_noise):
                    noise.sigma = max(args.sigma_min, noise.sigma * 0.99)
                logger.log_sigma(ou_noise[0].sigma)

        logger.push(global_step)
        if global_step % 9999 == 0:
            agent.save(f"models/{exp_name}/")

    logger.log_artifact()
    envs.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = DDPGLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
