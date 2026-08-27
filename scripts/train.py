# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np

from dylam.methods.sac import SAC, SACStrat, SACStratOpenLoop
from dylam.utils.experiment import get_experiment, make_env, parse_args, setup_run
from dylam.utils.logger import SACLogger


def train(args, exp_name, logger: SACLogger):
    if getattr(args, "scalar_critic", False):
        raise NotImplementedError(
            "scalar_critic (DyLam weights, single critic) is implemented for the "
            "tabular path only; see train_q_learning.py. The VSS config exists so the "
            "run is ready to launch once the SAC-side variant is written."
        )
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )
    if args.stratified:
        agent_cls = (
            SACStratOpenLoop if getattr(args, "open_loop_schedule", None) else SACStrat
        )
        agent = agent_cls(
            args,
            envs.single_observation_space,
            envs.single_action_space,
        )
    else:
        agent = SAC(args, envs.single_observation_space, envs.single_action_space)

    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(args.num_envs)]
            )
        else:
            actions = agent.get_action(obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards, terminations | truncations)

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
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
