# DyLam

Reinforcement-learning research code for DyLam (dynamic λ weighting over decomposed
reward components), plus the analysis pipeline that turns training runs into the
figures and tables of the DyLam manuscript.

## Language

### Experiments

**Component**:
One named term of a decomposed reward vector (e.g. `Ball_to_goal`, `Energy`).
Its name must match an entry in `comp_names`.
_Avoid_: objective, reward term, sub-reward

**Setup**:
A named method configuration resolved from `scripts/experiments.yml` — `Baseline`,
`Decq`, `Udc`, `Dylam`, `Dynmorl`.
_Avoid_: method, algorithm, variant

**Method**:
The paper-facing row label for a trad/ablation comparison (e.g. `Base SO RL`,
`UDC`, `Tuned-UDC`, `DyLam`, `DyLam-Scalar`). Distinct from Setup: one Setup
can be more than one Method depending on other config -- `Tuned-UDC` and
`UDC` are both the `Drq` setup, distinguished only by which wandb env their
runs were launched under.
_Avoid_: setup (see Setup), baseline, row label

**Nominal**:
The reference configuration a scope compares against — the published bounds for
robustness, the default hyperparameters for ablations. Every non-nominal cell is
measured against it.
_Avoid_: baseline (that word means the `Baseline` setup), default, control

**Condition**:
A perturbation of a setup's reward bounds, used as an experimental cell in the
robustness scope (e.g. the `±25%` bound misspecifications of RQ3).
_Avoid_: variant, config

**Arm**:
One cell of an ablation: a single hyperparameter moved off nominal with every
other ablated key pinned to nominal. A run matching only on the varied key is
not an arm — it may belong to a different cell of the sweep.
_Avoid_: ablation condition, sweep point, cell

**Seed summary**:
The mean of the final 10% of one run's logged metric history. The unit of
independence for every statistical test in the paper — one number per seed.
_Avoid_: final score, run result

**Family**:
A group of hypothesis tests sharing one Holm–Bonferroni correction (e.g. the RQ3
robustness family). Corrections never cross families.

### Analysis pipeline

**Paper artifact**:
A single figure or table that appears in the manuscript, identified by its LaTeX
label (`fig:res/robustness/curves`, `tab:res/trad/iqm`).
_Avoid_: output, plot, result

**Paper scope**:
A group of paper artifacts sharing a data source and a statistical treatment:
`trad`, `curriculum`, `robustness`, `morl`, `ablation`. The top-level unit of
organisation in `result_analysis/`.
_Avoid_: section, category, area

**Generator**:
The script that produces a paper artifact from tidy CSVs. Exactly one generator
owns each artifact.
_Avoid_: plotter, analysis script

**Tidy CSV**:
Committed per-seed long-form data. The only data shape a generator may
read. Default shape is one row per (step, seed), columns `_step,
<metric>, seed`, plus an identity column where a scope has more than one
condition/method/arm per file. A scope whose rows are Pareto candidate
points rather than a step series (morl) drops `_step` for `obj1..objN,
seed, point_index`, one file per source instead of an identity column.
_Avoid_: the CSVs, exports, data

**Aggregated export**:
Legacy wandb-UI download with `Step, mean, __MIN, __MAX` columns. Seeds are
unrecoverable from it, so it cannot feed a statistical test — only a curve.
_Avoid_: CSV export, wandb CSV

**Run cache**:
Ephemeral per-run wandb history keyed by run id, living outside the repo. A
network cache, never an archive — anything worth keeping becomes a tidy CSV.
_Avoid_: cache dir, data cache

### Beyond PDF

**Beyond PDF submission**:
The DyLam manuscript rendered as a web-native page for TMLR's Beyond PDF track,
built from the author kit's Jekyll/distill layout. A parallel presentation of the
same paper, not a separate publication.
_Avoid_: companion, website, blog post, HTML version

**Embed**:
A self-contained interactive HTML document iframed into the Beyond PDF
submission, produced by an Embed generator from tidy CSVs or checkpoints --
a wider read contract than Generator's, which is tidy-CSV-only. Distinct from
a Paper artifact: an Embed carries no LaTeX label, and one Embed commonly
supersedes several Paper artifacts.
_Avoid_: widget, applet, interactive figure
