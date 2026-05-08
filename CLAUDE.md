# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (Python ≥3.10, uv required)
uv venv && uv pip install .

# All scripts must be run from scripts/ (experiments.yml is resolved relative to CWD)
uv run python scripts/train.py --env LUNARLANDER --setup Dylam --track
uv run python scripts/train_dqn.py --env MINECART --setup Dynmorl
uv run python scripts/train_dynmorl.py --env MINECART --setup DynMORL --cuda false
uv run python scripts/train_q_learning.py --env CHICKENBANANA --setup Dylam
uv run python scripts/train_ppo.py --env LUNARLANDER --setup Dylam
uv run python scripts/test.py   # reads from models/to_test/

# Lint / format
uv run ruff check src/ scripts/
uv run isort src/ scripts/
```

CLI flags shared by all train scripts: `--env`, `--setup`, `--cuda`, `--capture-video`, `--video-freq`, `--track`, `--seed`.

## Architecture

### Experiment config
`scripts/experiments.yml` is the single source of truth for all hyperparams. Top-level keys are setup names: `Baseline`, `Decq`, `Udc`, `Dylam`, `Dynmorl`. Under each, environment keys (`MINECART`, `LUNARLANDER`, etc.) override defaults from `src/dylam/utils/experiment.py::base_hyperparams()`.

`--setup DynMORL` resolves to YAML key `Dynmorl` because `parse_args` applies `.lower().title()` to the input.

### Method selection pattern
Each `train_*.py` script creates a vectorised env via `gym.vector.AsyncVectorEnv` + `make_env`, instantiates the method class, runs the step loop, and delegates to the method's `get_action` / `update` / `save`. Method classes live in `src/dylam/methods/`.

- `DQN` / `DQ` / `UDC` — `dqn.py`. Baseline / stratified (one Q per reward) / unified double-critic variants.
- `DynMORL` — `dynmorl.py`. Weight-conditioned Q-network (`Q(s,w)`) with optional Diverse Experience Replay. `w` is drawn from `WeightSchedule` in `utils/weight_scheduler.py` and passed to both `get_action(obs, w)` and the replay buffer.
- SAC — `sac.py`. Continuous-action actor-critic.
- PPO — `ppo.py`. On-policy actor-critic using `PPOBuffer`.

### Networks
`src/dylam/methods/networks/architectures.py::QNetwork` is the shared backbone for all discrete-action methods. Its `forward(state, action, lambdas)` concatenates all three inputs. To condition a Q-net on weights `w`, set `num_inputs = obs_size + num_rewards`, `num_actions = 0`, and pass `w` as `lambdas` — **do not create a new network class**.

`TargetCritic` (in `targets.py`) wraps any `QNetwork`/`DoubleQNetwork` for Polyak-averaged target sync via `.sync(tau)`.

### DyLam λ update
`DQ.update_lambdas()` implements the DyLam weighting: exponential reweighting of per-objective λ based on recent mean rewards vs. `r_min`/`r_max` bounds. Gated by `args.dylam`. Uses `StratLastRewards` from `utils/buffer.py` as a rolling window.

### DynMORL DER buffer
`utils/diverse_buffer.py::DiverseReplayBuffer` extends `ReplayWeightAwareBuffer` with a secondary pool maintained by NSGA-II crowding distance over stored weight vectors. `sample` splits 50/50 between recent and diverse pools when secondary has ≥2 entries.

### Reward structure
Multi-objective envs return a vector reward. When `stratified: true`, `make_env` wraps with `MORecordEpisodeStatistics`; otherwise scalarised via `LinearReward`. Env wrappers in `src/dylam/envs/` track per-objective cumulative rewards in `info["reward_<name>"]` keys — these must match `comp_names` in the config.

### Logging
All loggers in `src/dylam/utils/logger.py` inherit `WandbResultLogger`. `log_episode` reads `info["Original_reward"]` and `info["reward_*"]` keys. `push(global_step)` flushes to wandb. Set `--track false` (default) to run in `wandb` disabled mode.

## Key constraints
- `comp_names`, `r_min`, `r_max` length must equal `num_rewards` — shape mismatch causes silent errors.
- Missing `--setup`/`--env` combo in `experiments.yml` → `KeyError` crash; always add both.
- `rsoccer-gym` (VSS env) only installs on Linux.
- `test.py` hardcodes `models/to_test/` as checkpoint directory.
