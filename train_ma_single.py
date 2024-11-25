# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np

from methods.sac import SAC, SACStrat
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run
from utils.logger import SACLogger


def train(args, exp_name, logger: SACLogger):
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )
    observation_space = envs.single_observation_space
    action_space = envs.single_action_space
    num_obs_per_agent = np.array(observation_space.shape).prod() // args.num_agents
    num_actions_per_agent = np.array(action_space.shape).prod() // args.num_agents
    single_agent_observation_space = gym.spaces.Box(
        low=observation_space.low[:num_obs_per_agent],
        high=observation_space.high[:num_obs_per_agent],
        shape=(num_obs_per_agent,),
        dtype=observation_space.dtype,
    )
    single_agent_action_space = gym.spaces.Box(
        low=action_space.low[:num_actions_per_agent],
        high=action_space.high[:num_actions_per_agent],
        shape=(num_actions_per_agent,),
        dtype=action_space.dtype,
    )
    if args.stratified:
        agent = SACStrat(
            args,
            single_agent_observation_space,
            single_agent_action_space,
        )
    else:
        agent = SAC(args, single_agent_observation_space, single_agent_action_space)
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        start = 0
        end = num_obs_per_agent
        multi_obs = []
        for _ in range(args.num_agents):
            multi_obs.append(obs[:, start:end])
            start = end
            end += num_obs_per_agent
        if global_step < args.learning_starts:
            multi_actions = np.array(
                [
                    [single_agent_action_space.sample() for _ in range(args.num_agents)]
                    for _ in range(args.num_envs)
                ]
            )
            actions = multi_actions.reshape((-1, action_space.shape[0]))
        else:
            multi_actions = np.array([agent.get_action(obs_) for obs_ in multi_obs])
            actions = np.concatenate(multi_actions, axis=1)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        logger.log_episode(infos, rewards)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        multi_next_obs = []
        start = 0
        end = num_obs_per_agent
        for _ in range(args.num_agents):
            multi_next_obs.append(real_next_obs[:, start:end])
            start = end
            end += num_obs_per_agent
        for i in range(args.num_agents):
            single_agent_rewards = rewards[:, : args.shared_rewards]
            single_agent_rewards = np.concatenate(
                (single_agent_rewards, rewards[:, args.shared_rewards + i].reshape(-1, 1)), axis=1
            )
            single_agent_action = actions[:, i * num_actions_per_agent : (i + 1) * num_actions_per_agent]
            agent.replay_buffer.add(
                multi_obs[i],
                single_agent_action,
                single_agent_rewards,
                multi_next_obs[i],
                terminations,
            )
        obs = next_obs

        if args.dylam:
            single_agent_rewards = rewards[:, : args.shared_rewards + 1]
            for i in range(1, args.num_agents):
                single_agent_rewards[:, -1] += rewards[:, args.shared_rewards + i]
            agent.add_episode_rewards(single_agent_rewards, terminations, truncations)
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


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
