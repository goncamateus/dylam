import time

import gymnasium as gym
import mo_gymnasium as mogym
import numpy as np

from methods.sac import SAC, SACStrat
from utils.experiment import get_experiment, make_env
from utils.experiment import parse_args
from utils.experiment import setup_run


def test(args):
    env = mogym.make(args.gym_id, render_mode="human")
    if args.stratified:
        env = mogym.MORecordEpisodeStatistics(env)
    else:
        env = mogym.LinearReward(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.stratified:
        agent = SACStrat(
            args,
            env.observation_space,
            env.action_space,
        )
    else:
        agent = SAC(args, env.observation_space, env.action_space)

    agent.load("models/to_test/")
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            obs = np.array([obs])
            actions = agent.get_action(obs)[0]
            next_obs, rewards, termination, truncated, infos = env.step(actions)
            done = termination or truncated
            obs = next_obs
        print(infos)


def main(params):
    setup_run(params)
    test(params)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
