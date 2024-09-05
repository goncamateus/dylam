# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import numpy as np

from methods.q_learning import QLearning, drQ, QDyLam
from utils.experiment import get_experiment, make_env
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
    env = make_env(args, 0, exp_name)()
    agent_type = get_agent_type(args)
    agent = agent_type(args, env.observation_space, env.action_space)

    obs, _ = env.reset()
    for global_step in range(args.total_timesteps):

        action = agent.get_action(obs)

        next_obs, reward, termination, truncation, info = env.step(action)
        logger.log_episode(info, reward)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncation):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]

        if args.dylam:
            agent.add_episode_rewards(reward, termination, truncation)
            agent.update_lambdas()
        
        agent.update_policy(obs, action, reward, next_obs)
        obs = next_obs
        # Falta a parte dos logs etc
        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and global_step % args.update_frequency == 0
        ):
            update_actor = global_step % args.policy_frequency == 0
            losses = agent.update(args.batch_size, update_actor)

            if global_step % args.target_network_frequency == 0:
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                loss_dict = {
                    "policy_loss": losses[0],
                    "qf1_loss": losses[1],
                    "qf2_loss": losses[2],
                    "alpha": agent.alpha,
                    "alpha_loss": losses[3],
                }
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
    logger = SACLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
