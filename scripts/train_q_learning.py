# Created by: Mateus Gonçalves Machado
# Based on: https://docs.cleanrl.dev/ (by Shengyi Huang)

import time

import numpy as np

from dylam.methods.q_learning import DQ, UDC, QDyLam, QDyLamScalar, QLearning
from dylam.utils.experiment import get_experiment, parse_args, q_make_env, setup_run
from dylam.utils.logger import QLogger


def get_agent_type(args):
    if getattr(args, "scalar_critic", False):
        if not args.dylam:
            raise ValueError("scalar_critic is a DyLam ablation; set dylam: True")
        return QDyLamScalar
    if args.stratified:
        if args.dylam:
            return QDyLam
        if args.realistic:
            return UDC
        return DQ
    return QLearning


def snapshot_path(exp_name, episode):
    return f"models/{exp_name}/snapshots/{episode:05d}/"


def save_snapshot(agent, path):
    """Snapshot component Q-tables and lambdas for beyond_pdf's export.

    agent.save() alone omits lambdas -- fine for the normal resume/inference
    path, not fine here: lambda at each moment of training is exactly what
    the beyond_pdf artifact displays.
    """
    agent.save(path)
    if hasattr(agent, "lambdas"):
        np.save(path + "lambdas.npy", agent.lambdas)


def train(args, exp_name, logger: QLogger):
    env = q_make_env(args, exp_name)
    agent_type = get_agent_type(args)
    agent = agent_type(args, env.observation_space, env.action_space)

    if args.checkpoint_interval:
        save_snapshot(agent, snapshot_path(exp_name, 0))

    for episode in range(1, args.total_episodes + 1):
        obs, _ = env.reset()
        termination = False
        truncation = False
        while not (termination or truncation):
            action = agent.get_action(obs)

            next_obs, reward, termination, truncation, info = env.step(action)
            if (termination or truncation) and args.strategy == 1:
                agent.epsilon_greedy_decay()

            info.update({"reward_epsilon": agent.epsilon})
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
            obs = next_obs

        if args.dylam:
            agent.update_lambdas()
            logger.log_lambdas(agent.lambdas)

        logger.push(episode)
        if episode % 10 == 0:
            agent.save(f"models/{exp_name}/")
        if args.checkpoint_interval and episode % args.checkpoint_interval == 0:
            save_snapshot(agent, snapshot_path(exp_name, episode))

    logger.log_artifact()
    env.close()


def main(params):
    gym_name = params.gym_id.split("-")[1]
    exp_name = f"{gym_name}-{params.setup}_{int(time.time())}"
    logger = QLogger(exp_name, params)
    setup_run(params)
    train(params, exp_name, logger)
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
