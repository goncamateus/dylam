# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import torch
import wandb

import envs

from methods.linear_support import LinearSupport
from methods.sac_gpi_ls import SACGPILS
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import SACLogger


def train(args, exp_name, logger: SACLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )

    agent = SACGPILS(args, envs.single_observation_space, envs.single_action_space)
    linear_support = LinearSupport(num_objectives=args.num_rewards, epsilon=None)

    obs, _ = envs.reset()
    global_step = 0
    for iteration in range(1, agent.max_iterations + 1):
        agent.set_weight_support(linear_support.get_weight_support())
        lambdas = linear_support.next_weight(
            algo="gpi-ls",
            gpi_agent=self,
            env=eval_env,
            rep_eval=num_eval_episodes_for_front,
        )
        print("Next weight vector:", lambdas)
        weight_support = (
            linear_support.get_weight_support()
            + linear_support.get_corner_weights(top_k=4)
            + [lambdas]
        )

        agent.set_weight_support(weight_support)
        tensor_weights = torch.tensor(weight_support, device=agent.device)
        for step in range(args.steps_per_iteration):
            if global_step < args.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(args.num_envs)]
                )
            else:
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

        for wcw in weight_support:
            n_value = policy_evaluation_mo(
                self, eval_env, wcw, rep=num_eval_episodes_for_front
            )[3]
            linear_support.add_solution(n_value, wcw)

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
