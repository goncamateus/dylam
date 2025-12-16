# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import gymnasium as gym
import numpy as np
import torch

from dylam.methods.ppo import PPO, PPOStrat
from dylam.utils.experiment import get_experiment, make_env, parse_args, setup_run
from dylam.utils.logger import PPOLogger


def train(args, exp_name, logger: PPOLogger):
    num_train_iterations = args.total_timesteps // args.num_steps
    print(f"Training for {num_train_iterations} iterations")
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, exp_name) for i in range(args.num_envs)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.stratified:
        agent = PPOStrat(
            args,
            envs.single_observation_space,
            envs.single_action_space,
        )
    else:
        agent = PPO(args, envs.single_observation_space, envs.single_action_space)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    state, _ = envs.reset(seed=args.seed)
    state = torch.Tensor(state).to(device)
    done = torch.zeros(args.num_envs).to(device)

    for train_iteration in range(1, num_train_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (train_iteration - 1.0) / num_train_iterations
            lr_now = args.learning_rate * frac
            agent.actor_optim.param_groups[0]["lr"] = lr_now
        for step in range(args.num_steps):
            global_step += 1

            # ALGO LOGIC: action logic
            with torch.no_grad():
                act, logprob, _ = agent.get_policy_returns(state)
                value = agent.get_value(state).squeeze()

            next_state, reward, terminations, truncations, infos = envs.step(act)
            agent.add_to_buffer(
                step,
                state,
                torch.Tensor(act).to(device),
                logprob,
                torch.tensor(reward).to(device),
                done,
                value,
            )
            if args.dylam:
                agent.add_episode_rewards(reward, terminations, truncations)
                agent.update_lambdas()
            done = np.logical_or(terminations, truncations)
            state = torch.Tensor(next_state).to(device)
            done = torch.Tensor(done).to(device)
            logger.log_episode(infos, reward, done)

        # ALGO LOGIC: training.
        loss_mean, policy_loss_mean, qf_loss_mean, entropy_loss_mean = agent.update(
            state, done
        )

        if global_step % 100 == 0:
            loss_dict = {}
            loss_dict["qf_loss"] = qf_loss_mean
            loss_dict["policy_loss"] = policy_loss_mean
            loss_dict["entropy_loss"] = entropy_loss_mean
            loss_dict["loss"] = loss_mean
            logger.log_losses(loss_dict)
        if args.dylam:
            logger.log_lambdas(agent.lambdas)
        logger.push(global_step)
        if global_step % 99 == 0:
            agent.save(f"models/{exp_name}/")
    logger.log_artifact()
    envs.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = PPOLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
