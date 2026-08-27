"""Self-check for the R5 open-loop lambda replay: the agent must follow the CSV.

Run:  python tests/test_open_loop_schedule.py
"""
import argparse
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

from dylam.methods.sac import SACStratOpenLoop

SCHEDULE = Path(__file__).parent.parent / "scripts/schedules/vss_dylam_lambda.csv"


def args_for(schedule):
    """Real config: base defaults plus the Dylam_Openloop VSS block from the yml."""
    from yaml import safe_load

    from dylam.utils.experiment import base_hyperparams

    params = base_hyperparams()
    with open(Path(__file__).parent.parent / "scripts/experiments.yml") as f:
        params.update(safe_load(f)["Dylam_Openloop"]["VSS"])
    params.update(dict(open_loop_schedule=str(schedule), cuda=False, num_envs=1,
                       buffer_size=1000, seed=0))
    return argparse.Namespace(**params)


def main():
    table = np.loadtxt(SCHEDULE, delimiter=",", skiprows=1)
    agent = SACStratOpenLoop(args_for(SCHEDULE), Box(-1, 1, (10,)), Box(-1, 1, (2,)))

    # the weights must come from the schedule, not from the returns: feed it returns
    # that would drive the closed-loop rule somewhere else entirely
    for _ in range(1000):
        agent.add_episode_rewards(
            np.array([[1e6, 1e6, 1e6]]), np.array([True]), np.array([False])
        )
    agent.update_lambdas()
    got = agent.lambdas.cpu().numpy()
    want = np.array([np.interp(1, table[:, 0], table[:, i]) for i in (1, 2, 3)])
    assert np.allclose(got, want, atol=1e-6), f"step 1: {got} != {want}"

    for target in (100_000, 250_000, 499_000):
        agent.open_loop_step = target - 1
        agent.update_lambdas()
        got = agent.lambdas.cpu().numpy()
        want = np.array([np.interp(target, table[:, 0], table[:, i]) for i in (1, 2, 3)])
        assert np.allclose(got, want, atol=1e-6), f"step {target}: {got} != {want}"

    # and the schedule must actually move, or the ablation tests nothing
    first = np.array([np.interp(0, table[:, 0], table[:, i]) for i in (1, 2, 3)])
    last = np.array([np.interp(5e5, table[:, 0], table[:, i]) for i in (1, 2, 3)])
    assert np.abs(first - last).max() > 0.05, "schedule is nearly constant"
    print("open-loop schedule replay OK")


if __name__ == "__main__":
    main()
