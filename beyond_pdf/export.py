"""Beyond PDF export: builds a single self-contained HTML file -- the
interactive lambda-simplex training scrubber for ChickenBanana/DyLam -- from
real per-episode checkpoints. See issue #38 for the full spec; this module
implements its "Export" and "Fidelity precondition" sections.

Reads only:
  - a directory of episode-numbered snapshot subdirectories, each holding
    components_q.npy (shape (num_rewards, n_states, n_actions)) and
    lambdas.npy (shape (num_rewards,)), written by
    scripts/train_q_learning.py's --checkpoint-interval flag;
  - beyond_pdf/template.html, the committed page shell this script injects
    one JSON data blob into -- the export is decoupled from the page's
    markup, so restyling the page never requires re-running this script;
  - optionally, a small actual-returns CSV (episode,<name0>,<name1>,<name2>)
    for the ghost-bar overlay (see beyond_pdf/extract_actual_returns.py).
    Omitting it drops the ghost bars, the first rung of the degradation
    ladder.
Touches no network and needs no wandb access.

Deterministic tie-breaking. QDyLam.get_output -- the method's own
greedy-action rule, called here unmodified -- breaks ties among
equal-valued actions with np.random. That is fine during training, where
the global RNG stream is whatever it is, but it is fatal to the dedup step
here: two weightings that are genuinely equivalent (identically-tied
Q-values at some state) would otherwise get arbitrarily different
tie-broken actions purely because they landed at different points in one
shared RNG stream, fragmenting the lattice into near-total uniqueness
instead of collapsing into the few real behaviours a converged run
actually has. The fix leaves get_output's code untouched and only manages
the RNG around each call: seed with the state index itself before every
call and restore the caller's RNG state after, so a tie at state s
resolves the same way every time s is tied, regardless of which episode or
weighting is asking. Verified empirically against the real ChickenBanana
run this was built from: this took the fully converged snapshot's
231-weighting lattice from 231 distinct policies (no collapsing at all) to
10.

Fidelity precondition. Before writing anything, this rolls out the last
snapshot's own components_q under its own final lambda and compares the
resulting per-Component returns against an expected-return target: either
--expected-return, given directly, or derived by --curriculum-data plus
--seed straight from the curriculum scope's committed tidy CSVs (avoids
transcribing those numbers by hand, a slip no automated check could
otherwise catch). On mismatch it aborts -- nonzero exit, a diagnostic on
stderr -- and writes no output file. This guarantee is attached to the
artifact rather than to a test, because the real checkpoints it compares
against are not in version control, and a test asserting on them would
silently skip on a fresh clone; see tests/test_beyond_pdf_export.py for how
the guard itself is tested (negatively, with a deliberately inconsistent
synthetic snapshot).

Usage:
  python beyond_pdf/export.py --snapshots PATH --out PATH \
      (--expected-return GATE BANANA CHICKEN | --curriculum-data DIR --seed N) \
      [--lattice-step 20] [--seed-criterion TEXT] [--actual-returns CSV]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "result_analysis"))
sys.path.insert(0, str(REPO_ROOT / "result_analysis" / "curriculum"))

import mo_gymnasium as mogym  # noqa: E402
import yaml  # noqa: E402
from core.style import COMPONENT_PALETTE  # noqa: E402
from sources import ENVS  # noqa: E402

import dylam  # noqa: E402,F401  registers mo-ChickenBanana-v0 on import
from dylam.methods.q_learning import QDyLam  # noqa: E402
from dylam.utils.experiment import base_hyperparams  # noqa: E402

TEMPLATE = Path(__file__).parent / "template.html"
EXPERIMENTS_YML = REPO_ROOT / "scripts" / "experiments.yml"
DATA_TOKEN = "/*__BEYOND_PDF_DATA__*/null"
SIZE_CAP_BYTES = 5 * 1024 * 1024
STEP_LIMIT = 80
BEHAVIOR_CLASSES = ["gate_only", "gate_banana", "gate_chicken", "gate_both", "no_gate"]
COMPONENTS = ENVS["CHICKENBANANA"].components  # (Objective/Gate, Banana, Chicken)
DEFAULT_SEED_CRITERION = (
    "median (lower of the two middle, ties broken by seed value) of the ten "
    "published seeds, ranked by the mean summed per-Component episode return "
    "over the final 10% of episodes, computed from the curriculum scope's "
    "committed tidy CSVs"
)


def build_agent(env):
    """A QDyLam sized for ChickenBanana, hyperparams read from the same
    scripts/experiments.yml block a real Dylam/CHICKENBANANA training run
    uses -- avoids a second, driftable copy of those numbers here. Only
    num_rewards/r_max/r_min actually affect get_output's inference-only
    codepath; the rest exist because QDyLam.__init__ requires them."""
    with open(EXPERIMENTS_YML) as f:
        config = yaml.safe_load(f)["Dylam"]["CHICKENBANANA"]
    params = base_hyperparams()
    params.update(config)
    params["lambdas"] = [1]
    args = argparse.Namespace(**params)
    return QDyLam(args, env.observation_space, env.action_space)


def lattice(step):
    """Barycentric lattice over the 3-Component simplex at 1/step resolution."""
    points = []
    for i in range(step + 1):
        for j in range(step + 1 - i):
            k = step - i - j
            points.append((i / step, j / step, k / step))
    return points


def greedy_policy(agent, weighting, n_states):
    """The full-grid greedy policy under one weighting -- see the module
    docstring for why ties are seeded by state index rather than left to the
    ambient RNG stream."""
    agent.lambdas = np.asarray(weighting, dtype=float)
    policy = np.empty(n_states, dtype=int)
    for s in range(n_states):
        rng_state = np.random.get_state()
        np.random.seed(s)
        policy[s] = agent.get_output(s)
        np.random.set_state(rng_state)
    return policy


def rollout(env, policy):
    """One greedy, exploration-free rollout from the start state."""
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    path = [[int(v) for v in unwrapped.agent_pos]]
    terminated = truncated = False
    steps = 0
    info = unwrapped.cumulative_reward_info
    while not (terminated or truncated) and steps < STEP_LIMIT:
        action = int(policy[obs])
        obs, _, terminated, truncated, info = env.step(action)
        path.append([int(v) for v in unwrapped.agent_pos])
        steps += 1
    returns = [float(info[f"reward_{c.name}"]) for c in COMPONENTS]
    if terminated:
        has_banana, has_chicken = unwrapped.has_banana, unwrapped.has_chicken
        if has_banana and has_chicken:
            behavior = "gate_both"
        elif has_banana:
            behavior = "gate_banana"
        elif has_chicken:
            behavior = "gate_chicken"
        else:
            behavior = "gate_only"
    else:
        behavior = "no_gate"
    return path, returns, behavior


def load_snapshot_episodes(snapshots_dir):
    return sorted(int(p.name) for p in Path(snapshots_dir).iterdir() if p.is_dir() and p.name.isdigit())


def load_snapshot(snapshots_dir, episode):
    d = Path(snapshots_dir) / f"{episode:05d}"
    components_q = np.load(d / "components_q.npy")
    lambdas = np.load(d / "lambdas.npy")
    return components_q, lambdas


def grid_metadata(env):
    unwrapped = env.unwrapped
    state_positions = [None] * unwrapped.n_states
    for (x, y), idx in unwrapped.obs_map.items():
        state_positions[idx] = [x, y]
    return {
        "layout": unwrapped.grid_layout,
        "start": [int(v) for v in unwrapped.agent_init_pos],
        "goal": list(unwrapped.goal_pos),
        "banana": list(unwrapped.banana_pos),
        "chicken": list(unwrapped.chicken_pos),
        "statePositions": state_positions,
    }


def expected_return_from_curriculum(curriculum_data, seed):
    """Read the fidelity target straight from the curriculum scope's committed
    tidy CSVs instead of requiring an author to transcribe it by hand into
    --expected-return -- a transcription slip there is a gap no automated
    check could otherwise catch."""
    values = []
    for c in COMPONENTS:
        path = Path(curriculum_data) / f"chickenbanana_{c.name.lower()}_reward.csv"
        rows = [r for r in csv.DictReader(open(path, newline="")) if int(r["seed"]) == seed]
        if not rows:
            sys.exit(f"no rows for seed {seed} in {path}")
        last = max(rows, key=lambda r: int(r["_step"]))
        values.append(float(last[f"ep_info/{c.name}"]))
    return values


def load_actual_returns(path, episodes, names):
    by_episode = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_episode[int(row["episode"])] = [float(row[n]) for n in names]
    return [by_episode.get(ep) for ep in episodes]


class DedupTable:
    """Collapses (episode, weighting) cells that share a policy into one
    entry, keyed on the concrete tie-broken policy array."""

    def __init__(self):
        self._index = {}
        self.entries = []

    def add(self, env, policy):
        key = tuple(policy.tolist())
        idx = self._index.get(key)
        if idx is None:
            path, returns, behavior = rollout(env, policy)
            idx = len(self.entries)
            self._index[key] = idx
            self.entries.append({"policy": list(key), "path": path,
                                  "returns": returns, "class": behavior})
        return idx


def check_fidelity(env, agent, snapshots_dir, final_episode, expected_return, tolerance):
    components_q, lambdas = load_snapshot(snapshots_dir, final_episode)
    agent.components_q = components_q
    n_states = env.observation_space.n
    policy = greedy_policy(agent, lambdas, n_states)
    _, returns, _ = rollout(env, policy)
    mismatch = max(abs(a - b) for a, b in zip(returns, expected_return))
    if mismatch > tolerance:
        sys.exit(
            "beyond_pdf fidelity check FAILED: rolling out the final snapshot "
            f"(episode {final_episode}) under its own final lambda "
            f"{lambdas.tolist()} gave returns {returns}, but --expected-return "
            f"was {list(expected_return)} (max abs diff {mismatch:.6g} > "
            f"tolerance {tolerance:.6g}). Aborting -- no output written. See "
            "the module docstring's RNG-drift note if this is a re-run of a "
            "seed that used to reproduce exactly."
        )
    return returns


def build(snapshots_dir, out_path, expected_return, lattice_step=20, tolerance=1e-6,
          seed=None, seed_criterion=None, actual_returns_path=None):
    episodes = load_snapshot_episodes(snapshots_dir)
    if not episodes:
        sys.exit(f"no snapshot episodes found under {snapshots_dir}")

    env = mogym.make("mo-ChickenBanana-v0")
    agent = build_agent(env)
    n_states = env.observation_space.n

    check_fidelity(env, agent, snapshots_dir, episodes[-1], expected_return, tolerance)

    weightings = lattice(lattice_step)
    dedup = DedupTable()
    cell_index = []
    real_index = []
    lambda_trajectory = []
    for episode in episodes:
        components_q, lambdas = load_snapshot(snapshots_dir, episode)
        agent.components_q = components_q
        lambda_trajectory.append(lambdas.tolist())

        row = [dedup.add(env, greedy_policy(agent, w, n_states)) for w in weightings]
        cell_index.append(row)
        real_index.append(dedup.add(env, greedy_policy(agent, lambdas, n_states)))
    env_grid = grid_metadata(env)
    env.close()

    actual_returns = None
    if actual_returns_path:
        actual_returns = load_actual_returns(actual_returns_path, episodes, [c.name for c in COMPONENTS])

    data = {
        "meta": {
            "env": "ChickenBanana-v0",
            "setup": "Dylam",
            "seed": seed,
            "seedCriterion": seed_criterion or DEFAULT_SEED_CRITERION,
            "episodes": episodes,
            "components": [
                {"name": c.name, "label": c.label, "rMax": c.r_max, "rMin": c.r_min,
                 "color": COMPONENT_PALETTE[i]}
                for i, c in enumerate(COMPONENTS)
            ],
            "latticeStep": lattice_step,
            "stepLimit": STEP_LIMIT,
            "nStates": int(n_states),
            "grid": env_grid,
            "behaviorClasses": BEHAVIOR_CLASSES,
        },
        "lambdaTrajectory": lambda_trajectory,
        "realIndex": real_index,
        "actualReturns": actual_returns,
        "lattice": [list(w) for w in weightings],
        "cellIndex": cell_index,
        "policies": dedup.entries,
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        sys.exit(f"template {TEMPLATE} is missing the data token {DATA_TOKEN!r}")
    html = template.replace(DATA_TOKEN, json.dumps(data, separators=(",", ":")))

    size = len(html.encode("utf-8"))
    if size > SIZE_CAP_BYTES:
        sys.exit(
            f"built artifact is {size} bytes, over the {SIZE_CAP_BYTES}-byte cap. "
            "Apply the degradation ladder in order: drop the ghost bars "
            "(--actual-returns), coarsen --lattice-step from 20 to 15, then drop "
            "the counterfactual dimension entirely."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({size} bytes, {len(dedup.entries)} distinct policies "
          f"over {len(episodes)} episodes x {len(weightings)} weightings)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", type=Path, required=True,
                     help="directory of episode-numbered snapshot subdirectories")
    ap.add_argument("--out", type=Path, required=True, help="output HTML path")
    ap.add_argument("--expected-return", type=float, nargs=3, default=None,
                     metavar=("GATE", "BANANA", "CHICKEN"),
                     help="fidelity precondition target (see module docstring). If "
                     "omitted, --curriculum-data and --seed derive it directly from "
                     "the curriculum scope's committed tidy CSVs instead of requiring "
                     "it to be transcribed by hand.")
    ap.add_argument("--curriculum-data", type=Path, default=None,
                     help="result_analysis/curriculum/data (or equivalent) to derive "
                     "--expected-return from, keyed by --seed")
    ap.add_argument("--lattice-step", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=None, help="informational only, shown on the page")
    ap.add_argument("--seed-criterion", type=str, default=None)
    ap.add_argument("--actual-returns", type=Path, default=None,
                     help="episode,<name0>,<name1>,<name2> CSV for the ghost-bar overlay")
    args = ap.parse_args()

    expected_return = args.expected_return
    if expected_return is None:
        if args.curriculum_data is None or args.seed is None:
            ap.error("--expected-return, or --curriculum-data together with --seed, is required")
        expected_return = expected_return_from_curriculum(args.curriculum_data, args.seed)

    build(args.snapshots, args.out, expected_return, args.lattice_step,
          args.tolerance, args.seed, args.seed_criterion, args.actual_returns)


if __name__ == "__main__":
    main()
