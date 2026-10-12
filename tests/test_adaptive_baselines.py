"""Self-check for the R5 adaptive baselines (ticket 10, ADR 0003).

Primal-dual: mu integrates the normalized violation, grows while a bound is missed,
relaxes to 0 once it is met, and the weights mu/sum(mu) stay on the simplex.
Linear: the hand-designed CSVs go from uniform to the env's native scalarization.
DyLam: the smoothed-returns refactor leaves its weights unchanged.

Run:  python tests/test_adaptive_baselines.py   (or pytest)
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from yaml import safe_load

from dylam.envs.chicken_banana import ChickenBanana
from dylam.methods.q_learning import QDyLamLagrangian, QDyLamOpenLoop
from dylam.methods.sac import SACStrat, SACStratLagrangian
from dylam.utils.experiment import base_hyperparams

ROOT = Path(__file__).parent.parent
with open(ROOT / "scripts/experiments.yml") as f:
    CONFIG = safe_load(f)


def args_for(setup, env, **extra):
    params = base_hyperparams()
    params.update(CONFIG[setup][env])
    params.update(dict(cuda=False, num_envs=1, buffer_size=1000, seed=0), **extra)
    return argparse.Namespace(**params)


def feed_sac(agent, returns, n):
    for _ in range(n):
        agent.add_episode_rewards(
            np.array([returns], dtype=float), np.array([True]), np.array([False])
        )
        agent.update_lambdas()


def test_sac_dual_grows_under_violation():
    args = args_for("Dylam_Lagrangian", "HALFCHEETAH")
    assert np.isclose(args.dual_lr, 1 - args.dylam_tau), "eta must be 1 - tau_lambda"
    agent = SACStratLagrangian(args, Box(-1, 1, (10,)), Box(-1, 1, (2,)))
    assert torch.allclose(agent.lambdas, torch.full((2,), 0.5)), "starts uniform"

    # both bounds missed (run 400 < 800, ctrl -500 < -200): every mu must grow
    feed_sac(agent, [400.0, -500.0], args.dylam_rb)
    before = agent.mu.clone()
    feed_sac(agent, [400.0, -500.0], 1000)
    assert (agent.mu > before).all(), f"mu did not grow: {before} -> {agent.mu}"
    assert torch.isclose(agent.lambdas.sum(), torch.tensor(1.0))
    # equal normalized violation (0.5 each) => equal weights
    assert torch.allclose(agent.lambdas, torch.full((2,), 0.5), atol=1e-3)


def test_tabular_dual_relaxes_when_satisfied():
    args = args_for("Dylam_Lagrangian", "CHICKENBANANA")
    agent = QDyLamLagrangian(args, Discrete(10), Discrete(4))

    def feed(returns, n):
        for _ in range(n):
            agent.add_episode_reward(np.array(returns, dtype=float), True, False)
            agent.update_lambdas()

    r_max = np.array(args.r_max, dtype=float)
    # Chicken missed (0 < R_max), Gate and Banana exactly at R_max: Chicken's mu grows,
    # the met ones hold their value. CB's R_max are the algebraic maxima, so this
    # freeze is all a real CB run can show; it is the integral term's memory.
    feed(r_max * [1, 1, 0], 3000)
    assert agent.mu[2] > 1 and np.allclose(agent.mu[:2], 1), agent.mu
    assert agent.lambdas[2] > agent.lambdas[0], agent.lambdas

    # overshoot (unreachable in CB, unit check of the projection): every mu relaxes
    # to 0 and the weights stay finite on the simplex
    feed(2 * r_max, 50_000)
    assert (agent.mu == 0).all(), agent.mu
    assert np.isfinite(agent.lambdas).all() and np.isclose(agent.lambdas.sum(), 1)


def test_linear_schedules_hit_endpoints():
    native = {"HALFCHEETAH": [16 / 16.6, 0.6 / 16.6], "VSS": [0.38, 0.52, 0.10]}
    for env, want_end in native.items():
        cfg = CONFIG["Dylam_Linear"][env]
        path = ROOT / "scripts" / cfg["open_loop_schedule"]
        with open(path) as f:
            assert f.readline().strip().split(",") == ["step"] + cfg["comp_names"]
        table = np.loadtxt(path, delimiter=",", skiprows=1)
        n = len(cfg["comp_names"])
        assert table[0, 0] == 0 and table[-1, 0] == cfg["total_timesteps"]
        assert np.allclose(table[0, 1:], 1 / n, atol=1e-4), f"{env} start not uniform"
        assert np.allclose(table[-1, 1:], want_end, atol=1e-4), f"{env} end wrong"


def test_tabular_open_loop_follows_schedule_per_episode(tmp_path=None):
    import tempfile

    path = Path(tmp_path or tempfile.mkdtemp()) / "cb.csv"
    path.write_text("step,Gate,Banana,Chicken\n0,0.2,0.3,0.5\n100,0.6,0.3,0.1\n")
    args = args_for("Dylam", "CHICKENBANANA", open_loop_schedule=str(path))
    agent = QDyLamOpenLoop(args, Discrete(10), Discrete(4))
    assert np.allclose(agent.lambdas, [0.2, 0.3, 0.5])
    for _ in range(50):  # returns that the closed-loop rule would act on are ignored
        agent.add_episode_reward(np.zeros(3), True, False)
        agent.update_lambdas()
    assert np.allclose(agent.lambdas, [0.4, 0.3, 0.3]), agent.lambdas


def test_chicken_banana_rewards_are_binary():
    env = ChickenBanana()
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.reset(seed=int(rng.integers(1 << 30)))
        gate, done = 0.0, False
        while not done:
            _, r, term, trunc, info = env.step(int(rng.integers(4)))
            assert set(np.unique(r)) <= {0.0, 1.0}, r
            gate += r[0]
            done = term or trunc
        assert info["reward_Objective"] == gate, (info, gate)
        assert info["Original_reward"] <= 3


def test_dylam_weights_unchanged_by_refactor():
    args = args_for("Dylam", "HALFCHEETAH")
    agent = SACStrat(args, Box(-1, 1, (10,)), Box(-1, 1, (2,)))
    r_max, r_min = np.array(args.r_max), np.array(args.r_min)
    window, g = deque(maxlen=args.dylam_rb), None
    for ret in ([300.0, -300.0], [500.0, -250.0]):
        feed_sac(agent, ret, args.dylam_rb)
        # reference: the pre-refactor rule, on an independent copy of the window
        for _ in range(args.dylam_rb):
            window.append(ret)
            if len(window) == args.dylam_rb:
                m = np.mean(window, axis=0)
                g = m if g is None else m + (g - m) * args.dylam_tau
    zeta = np.clip((r_max - g) / (r_max - r_min), 0, 1)
    want = agent.normalizer(torch.Tensor(zeta)).numpy()
    assert np.allclose(agent.lambdas.numpy(), want, atol=1e-5), (agent.lambdas, want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("adaptive baselines OK")
