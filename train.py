# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import wandb

import envs

from pyvirtualdisplay import Display

from methods.ddpg import DDPG
from methods.sac import SAC, SACStrat
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import DDPGLogger, SACLogger, WandbResultLogger


def train(args, exp_name, logger: SACLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )
    if args.stratified:
        agent = SACStrat(
            args,
            envs.single_observation_space,
            envs.single_action_space,
            hidden_dim=args.hidden_dim,
        )
    else:
        method = SAC if args.method == "sac" else DDPG
        agent = method(args, envs.single_observation_space, envs.single_action_space)

    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(args.num_envs)]
            )
        else:
            actions = agent.get_action(obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards)
        if terminations.any() or truncations.any():
            if agent.ou:
                agent.ou.reset()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(obs, actions, rewards, real_next_obs, terminations)
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if args.dylam:
                agent.add_episode_rewards(rewards, terminations, truncations)
                agent.update_lambdas()
            update_actor = global_step % args.policy_frequency == 0
            policy_loss, qf1_loss, qf2_loss, alpha_loss = agent.update(
                args.batch_size, update_actor
            )

            if global_step % args.target_network_frequency == 0:
                agent.critic_target.sync(args.tau)

            if global_step % 100 == 0:
                logger.log_losses(
                    {
                        "policy_loss": policy_loss,
                        "qf1_loss": qf1_loss,
                        "qf2_loss": qf2_loss,
                        "alpha": agent.alpha,
                        "alpha_loss": alpha_loss,
                    }
                )
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
    _display = Display(visible=0, size=(1400, 900))
    _display.start()
    logger = SACLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
