# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import torch

from dylam.methods.dynmorl import DynMORL
from dylam.utils.experiment import get_experiment, make_env, parse_args, setup_run
from dylam.utils.logger import DynMORLLogger
from dylam.utils.weight_scheduler import WeightSchedule


def train(args, exp_name, logger: DynMORLLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )

    agent = DynMORL(args, envs.single_observation_space, envs.single_action_space)

    schedule = WeightSchedule(
        args.total_timesteps,
        args.num_rewards,
        weight_change_freq=args.weight_change_freq,
        mode=args.weight_mode,
        seed=args.seed,
    )

    obs, _ = envs.reset()
    for global_step in range(1, args.total_timesteps + 1):
        w = schedule.current_w(global_step)
        actions = agent.get_action(obs, w)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards, terminations | truncations)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(
            obs, actions, rewards, real_next_obs, terminations,
            np.tile(w, (args.num_envs, 1)),
        )
        obs = next_obs

        if args.reset_optimizer_on_w_change and schedule.changed(global_step):
            agent.reset_optimizer()

        # ALGO LOGIC: training.
        if (
            global_step > args.learning_starts
            and global_step % args.update_frequency == 0
        ):
            losses = agent.update(args.batch_size)

            if global_step % args.target_network_frequency == 0:
                agent.target_q_network.sync(args.tau)

            if global_step % 100 == 0:
                loss_dict = {f"qf_loss_{i}": losses[i] for i in range(args.num_rewards)}
                logger.log_losses(loss_dict)
                logger.log_weights(torch.tensor(w))

        logger.push(global_step)
        if global_step % 9999 == 0:
            agent.save(f"models/{exp_name}/")
    agent.save(f"models/{exp_name}/")
    logger.log_artifact()
    envs.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = DynMORLLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
