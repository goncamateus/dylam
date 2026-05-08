# DyLam — Agent Operating Notes

## Setup
- `uv venv && uv pip install .` (Python ≥3.10)
- Runs from repo root — `experiments.yml` is loaded with a relative path from CWD.
- GPU required for meaningful runs (CUDA default; set `--cuda false` to test on CPU).

## Running experiments
All scripts live in `scripts/`. Config for every run lives in `scripts/experiments.yml`.

```
python scripts/train.py          # SAC
python scripts/train_dqn.py      # DQN / DQ / UDC
python scripts/train_q_learning.py  # Q-learning / DQ / UDC / QDyLam
python scripts/train_ppo.py      # PPO
python scripts/train_dynmorl.py  # DynMORL (weight-conditioned MO Q-network + DER)
python scripts/ablation.py       # Q-learning ablation sweeps (tau, rb, normalizer, …)
python scripts/test.py           # Evaluate a saved model from models/to_test/
```

CLI flags: `--env LUNARLANDER --setup Baseline --cuda --capture-video --track`
- `--env` maps to the key in experiments.yml (case-insensitive).
- `--setup` picks the top-level config block: `Baseline`, `Decq`, `Udc`, `Dylam`, or `DynMORL`.
- `--track` logs to wandb. `--capture-video` saves episode videos to `videos/`.
- `--seed` defaults to 0; ablation.py overrides with `int(time.time())` per sub-run.

## Config: experiments.yml
- Top-level keys: `Baseline`, `Decq`, `Udc`, `Dylam`, `Dynmorl`.
- Under each: environment keys like `CHICKENBANANA`, `LUNARLANDER`, `VSS`, `MINECART`, `HALFCHEETAH`.
- `gym_id` must match `mo-<Name>-v<version>` format for multi-objective envs.
- `dylam: true` enables DyLam's dynamic reward weighting (uses `r_min`, `r_max`, `num_rewards`, `comp_names`, `normalizer`, `dylam_rb`, `dylam_tau`).
- DynMORL-specific keys: `algorithm` (`cond`/`mo`/`scal`/`uvfa`), `memory_strategy` (`std`/`der`), `weight_change_freq`, `weight_mode` (`regular`/`sparse`), `reset_optimizer_on_w_change`, `der_secondary_size`.
- `stratified: true` uses MO algorithms (one Q-network per reward); `realistic: true` selects UDC method within stratified.
- Default hyperparams live in `src/dylam/utils/experiment.py::base_hyperparams()` and are overridden per-setup/env.

## Architecture
- **methods** (`src/dylam/methods/`): SAC, DQN, Q-learning, PPO — each has a baseline and a `Stratified` variant. `dynmorl.py` adds a weight-conditioned MO Q-network (DynMORL).
- **envs** (`src/dylam/envs/`): custom wrappers for Taxi, LunarLander, HalfCheetah, ChickenBanana, VSS, Minecart.
- **utils** (`src/dylam/utils/`): experiment config parsing, replay buffer, wrappers, loggers. `diverse_buffer.py` = DER two-pool buffer; `weight_scheduler.py` = DynMORL weight schedule iterator.
- **scripts/**: all entry points; `experiments.yml` is the single source of truth for hyperparams.
- **result_analysis/**: Jupyter notebooks for post-hoc analysis; `utils.py` shared across notebooks.

## Gotchas
- No `--setup` + `--env` combo exists in `experiments.yml` → script crashes with a KeyError. Always verify the pair exists.
- `ablation.py` runs nested loops (tau → rb → epsilon_decay → normalizer → tau_vs_epsilon), each with 10 seeds. Expect it to take a long time.
- `test.py` hardcodes `models/to_test/` as the checkpoint directory.
- `rsoccer-gym` dependency only installs on Linux (git source).
- Videos are only captured for env index 0; `--video-freq` controls episode sampling.
- Models saved every 9999 steps (SAC/PPO/DQN) or every 10 episodes (Q-learning).
- `comp_names` and `r_min`/`r_max` must be aligned with `num_rewards` — mismatch causes silent shape errors.

## Analysis
- Run notebooks from `result_analysis/` after experiments complete.
- Wandb artifacts stored when `--track` is used.
- Local outputs: `models/<exp_name>/` for checkpoints, `videos/<exp_name>/` for recordings, `wandb/` for local wandb data.
