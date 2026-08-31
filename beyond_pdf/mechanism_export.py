"""Beyond PDF export: builds a single self-contained HTML file -- the
mechanism explainer Embed (issue #47) -- from real ChickenBanana/Dylam
checkpoints and the committed per-episode actual-returns CSV (issue #38's
`--actual-returns`, beyond_pdf/data/chickenbanana_actual_returns.csv).

One worked window of E consecutive episodes, ending at --window-end, walks
through DyLam's weight update in six beats: (1) the per-Component rewards
over a context of recent episodes; (2) the sliding window of size E
highlighted over it; (3) an arrow to the window mean feeding the EMA update;
(4) the plot dissolving to leave the resolved smoothed-return/weight
equation; (5) the resulting lambda weights; (6) the greedy policy those
weights induce, reusing beyond_pdf/export.py's grid renderer.

Reads only:
  - a directory of episode-numbered snapshot subdirectories (the same shape
    beyond_pdf/export.py reads: components_q.npy, lambdas.npy per episode),
    for beat 6's policy;
  - beyond_pdf/data/chickenbanana_actual_returns.csv (committed), for beats
    1-5's per-episode rewards;
  - scripts/experiments.yml's Dylam/CHICKENBANANA block, for E, tau_lambda,
    epsilon, and the component bounds -- the same hyperparameters a real
    training run used, avoiding a second, driftable copy of those numbers;
  - beyond_pdf/mechanism_template.html, the committed page shell.
Touches no network and needs no wandb access.

Computing the smoothed return Gbar_i at --window-end. Checkpoints store the
resulting components_q and lambdas per episode, not the intermediate
smoothed-return state Eq. dylam-ema accumulates, so this module recomputes
Gbar_i by replaying that exact update, once per episode, over the full
actual-returns history from episode 1 to --window-end -- the same
deterministic formula and the same per-episode cadence the tabular training
loop used (Algorithm 1), from the same recorded per-episode returns. This is
recomputation of a deterministic quantity from real data, not simulation:
Eq. dylam-weights then maps the recomputed Gbar_i to a lambda, which this
module cross-checks against the checkpoint's own recorded lambdas.npy at
--window-end and aborts if they disagree by more than --tolerance --
the same fidelity-precondition idea export.py's module docstring documents,
applied here to catch a hyperparameter or replay bug rather than RNG drift.

Usage:
  python beyond_pdf/mechanism_export.py --snapshots PATH --out PATH \
      --window-end EPISODE [--actual-returns CSV] [--tolerance 1e-6]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "curriculum"))
sys.path.insert(0, str(Path(__file__).parent))

import mo_gymnasium as mogym  # noqa: E402
from core.style import COMPONENT_PALETTE  # noqa: E402
from export import (build_agent, grid_metadata, load_snapshot,  # noqa: E402
                    load_snapshot_episodes, rollout)
from sources import ENVS  # noqa: E402

from dylam.utils.experiment import base_hyperparams  # noqa: E402

TEMPLATE = Path(__file__).parent / "mechanism_template.html"
DATA_TOKEN = "/*__BEYOND_PDF_DATA__*/null"
SIZE_CAP_BYTES = 5 * 1024 * 1024
ACTUAL_RETURNS = Path(__file__).parent / "data" / "chickenbanana_actual_returns.csv"
COMPONENTS = ENVS["CHICKENBANANA"].components  # (Objective/Gate, Banana, Chicken)
EPSILON = 1e-4  # matches the mechanism section's Eq. dylam-weights ($\epsilon = 10^{-4}$)


def load_actual_returns(path):
    """episode (1-indexed, contiguous) -> {component name: G^(k)_i}."""
    by_episode = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_episode[int(row["episode"])] = {c.name: float(row[c.name]) for c in COMPONENTS}
    return by_episode


def dylam_hyperparams():
    """E, tau_lambda for a real Dylam/CHICKENBANANA run: base_hyperparams()'s
    defaults overridden by scripts/experiments.yml's Dylam/CHICKENBANANA
    block, exactly as export.py's build_agent sizes its agent -- the same
    numbers a real training run used, not a second copy."""
    with open(REPO_ROOT / "scripts" / "experiments.yml") as f:
        config = yaml.safe_load(f)["Dylam"]["CHICKENBANANA"]
    params = base_hyperparams()
    params.update(config)
    return params["dylam_rb"], params["dylam_tau"]


def replay_smoothed_returns(returns_by_episode, window_end, buffer_len, tau_lambda):
    """Replay Eq. dylam-window + Eq. dylam-ema once per episode from episode
    1 to `window_end`, exactly as Algorithm 1 initializes and updates them
    (ring buffer of the last `buffer_len` completed episodes; Gbar_i <- 0
    initially). Returns (Gbar at window_end, M at window_end, the window's
    own per-episode returns)."""
    names = [c.name for c in COMPONENTS]
    gbar = {n: 0.0 for n in names}
    episodes = sorted(returns_by_episode)
    if window_end not in returns_by_episode:
        sys.exit(f"--window-end {window_end} has no row in the actual-returns CSV")
    if window_end < buffer_len:
        sys.exit(f"--window-end {window_end} is before the buffer fills (E={buffer_len})")

    m_at_end, window_returns = None, None
    for k in episodes:
        if k > window_end:
            break
        if k < buffer_len:
            continue
        window = [returns_by_episode[j] for j in range(k - buffer_len + 1, k + 1)]
        m = {n: sum(w[n] for w in window) / buffer_len for n in names}
        gbar = {n: tau_lambda * gbar[n] + (1 - tau_lambda) * m[n] for n in names}
        if k == window_end:
            m_at_end, window_returns = m, window
    return gbar, m_at_end, window_returns


def dylam_weights(gbar, epsilon=EPSILON):
    rho, zeta, w = {}, {}, {}
    for c in COMPONENTS:
        rho[c.name] = (gbar[c.name] - c.r_min) / (c.r_max - c.r_min)
        zeta[c.name] = min(max(1 - rho[c.name], 0.0), 1.0)
        w[c.name] = float(np.exp(zeta[c.name]) - 1)
    total = sum(w.values())
    lam = {n: (w[n] + epsilon) / (total + epsilon) for n in w}
    return rho, zeta, w, lam


def nearest_snapshot_episode(snapshots_dir, window_end):
    episodes = load_snapshot_episodes(snapshots_dir)
    if not episodes:
        sys.exit(f"no snapshot episodes found under {snapshots_dir}")
    ge = [e for e in episodes if e >= window_end]
    return min(ge) if ge else max(episodes)


def build(snapshots_dir, out_path, window_end, actual_returns_path=ACTUAL_RETURNS,
          tolerance=1e-6, context_episodes=None):
    buffer_len, tau_lambda = dylam_hyperparams()
    context_episodes = context_episodes or max(3 * buffer_len, 30)

    returns_by_episode = load_actual_returns(actual_returns_path)
    gbar, m, window_returns = replay_smoothed_returns(
        returns_by_episode, window_end, buffer_len, tau_lambda)
    rho, zeta, w, lam = dylam_weights(gbar)

    snap_episode = nearest_snapshot_episode(snapshots_dir, window_end)
    components_q, checkpoint_lambdas = load_snapshot(snapshots_dir, snap_episode)
    if snap_episode == window_end:
        computed = np.array([lam[c.name] for c in COMPONENTS])
        mismatch = float(np.max(np.abs(computed - checkpoint_lambdas)))
        if mismatch > tolerance:
            sys.exit(
                "beyond_pdf mechanism export fidelity check FAILED: replaying the EMA "
                f"update to episode {window_end} gives lambda {computed.tolist()}, but "
                f"the checkpoint at that episode recorded {checkpoint_lambdas.tolist()} "
                f"(max abs diff {mismatch:.6g} > tolerance {tolerance:.6g}). Aborting -- "
                "no output written. Check --window-end lines up with scripts/"
                "experiments.yml's current Dylam/CHICKENBANANA E and tau_lambda."
            )

    env = mogym.make("mo-ChickenBanana-v0")
    agent = build_agent(env)
    agent.components_q = components_q
    agent.lambdas = checkpoint_lambdas
    n_states = env.observation_space.n
    policy = np.empty(n_states, dtype=int)
    for s in range(n_states):
        rng_state = np.random.get_state()
        np.random.seed(s)
        policy[s] = agent.get_output(s)
        np.random.set_state(rng_state)
    path, returns, behavior = rollout(env, policy)
    grid = grid_metadata(env)
    env.close()

    episodes = sorted(e for e in returns_by_episode if window_end - context_episodes < e <= window_end)
    context = {c.name: [returns_by_episode[e][c.name] for e in episodes] for c in COMPONENTS}

    data = {
        "meta": {
            "env": "ChickenBanana-v0", "setup": "Dylam",
            "bufferLen": buffer_len, "tauLambda": tau_lambda,
            "windowEnd": window_end, "snapshotEpisode": snap_episode,
            "components": [
                {"name": c.name, "label": c.label, "rMax": c.r_max, "rMin": c.r_min,
                 "color": COMPONENT_PALETTE[i]}
                for i, c in enumerate(COMPONENTS)
            ],
            "nStates": int(n_states),
            "grid": grid,
        },
        "context": {"episodes": episodes, "rewards": context},
        "window": {"episodes": list(range(window_end - buffer_len + 1, window_end + 1)),
                   "rewards": {c.name: [w[c.name] for w in window_returns] for c in COMPONENTS}},
        "m": m, "gbar": gbar, "rho": rho, "zeta": zeta, "w": w, "lambda": lam,
        "policy": {"policy": policy.tolist(), "path": path, "returns": returns, "class": behavior},
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        sys.exit(f"template {TEMPLATE} is missing the data token {DATA_TOKEN!r}")
    html = template.replace(DATA_TOKEN, json.dumps(data, separators=(",", ":")))

    size = len(html.encode("utf-8"))
    if size > SIZE_CAP_BYTES:
        sys.exit(f"built artifact is {size} bytes, over the {SIZE_CAP_BYTES}-byte cap.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({size} bytes, window ending episode {window_end}, "
          f"snapshot episode {snap_episode})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", type=Path, required=True,
                     help="directory of episode-numbered snapshot subdirectories")
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument("--window-end", type=int, required=True,
                     help="episode ending the E-episode window shown in beats 1-5")
    ap.add_argument("--actual-returns", type=Path, default=ACTUAL_RETURNS)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--context-episodes", type=int, default=None,
                     help="episodes of context shown before the window (default: 3E or 30)")
    args = ap.parse_args()
    build(args.snapshots, args.out, args.window_end, args.actual_returns,
          args.tolerance, args.context_episodes)


if __name__ == "__main__":
    main()
