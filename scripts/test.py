import gymnasium as gym
import mo_gymnasium as mogym
import numpy as np

from dylam.methods.sac import SAC, SACStrat
from dylam.utils.experiment import get_experiment, parse_args, setup_run


def test(args):
    env = mogym.make(args.gym_id)
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
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            obs = np.array([obs])
            actions = agent.get_action(obs)[0]
            next_obs, rewards, termination, truncated, infos = env.step(actions)
            done = termination or truncated
            obs = next_obs


def main(params):
    setup_run(params)
    test(params)


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    main(params)
