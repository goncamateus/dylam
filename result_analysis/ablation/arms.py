"""Canonical arm metadata for the Chicken--Banana ablation sweeps
(sections/ablation.tex, app:ablation).

Each ablation type varies exactly one of DyLam's hyperparameters off
nominal (tau_lambda=0.995, buffer E=10, epsilon-decay=0.9988, normalizer=
softmax) with a dedicated wandb setup per swept value. The manuscript
shows a subset of what was actually run (e.g. 3 of 5 swept tau values);
ARMS lists exactly that published subset, not everything in wandb.

CONFOUNDED: scripts/ablation.py sweeps tau, then rb, then epsilon-decay,
then normalizer, reusing one mutable params object across all four loops
without resetting the previous loop's field. Verified against the actual
logged wandb configs: every DyLam-RB-* run was trained with dylam_tau=0.5
(the tau sweep's last value) instead of nominal 0.995, and every
DyLam-EpsilonDecayFactor-* run inherited both that stale tau and
dylam_rb=500 (the rb sweep's last value). The normalizer sweep was run
separately, later, and is clean. This is a training-code bug (out of
scope for this analysis migration to fix); fig:ablation/rb and
fig:ablation/epsilon are reproduced faithfully from what wandb actually
contains, which is not what their captions describe.
"""
from collections import namedtuple

Arm = namedtuple("Arm", "label wandb_setup")

NOMINAL_SETUP = "Dylam"  # tau=0.995, rb=10, epsilon_decay=0.9988, normalizer=softmax

TAU_ARMS = [Arm("0.5", "DyLam-Tau-0.5"), Arm("0.7", "DyLam-Tau-0.7"), Arm("0.9", "DyLam-Tau-0.9")]

# CONFOUNDED (see module docstring): every run below has dylam_tau=0.5, not nominal.
RB_ARMS = [Arm("50", "DyLam-RB-50"), Arm("100", "DyLam-RB-100"), Arm("500", "DyLam-RB-500")]

# CONFOUNDED (see module docstring): every run below has dylam_tau=0.5, dylam_rb=500.
EPSILON_ARMS = [Arm("0.8", "DyLam-EpsilonDecayFactor-0.8"),
                Arm("0.9", "DyLam-EpsilonDecayFactor-0.9"),
                Arm("0.95", "DyLam-EpsilonDecayFactor-0.95")]

# fig:ablation/normalizer plots only these two non-nominal transforms; the
# corrected min-max variant (DyLam-Normalizer-minmax-fixed) is table-only.
NORMALIZER_ARMS = [Arm("l1", "DyLam-Normalizer-l1"), Arm("minmax", "DyLam-Normalizer-minmax")]

# tab:ablation/normalizer's four rows. Verified against wandb (not just
# setup-name spelling): "minmax" is the *reversed*-routing run kept as a
# sign-flip control (minmax_norm in src/dylam/utils/experiment.py maps the
# largest deficiency to the smallest weight); "minmax-fixed" is the
# corrected transform actually called "Min--max" in the table. Both were
# confirmed by reproducing each row's mean/std/success count before this
# mapping was trusted.
NORMALIZER_TABLE_ROWS = [
    Arm("Exponential, $g(\\zeta) = \\mathrm{e}^\\zeta - 1$", "DyLam-Normalizer-softmax"),
    Arm("Linear ($\\ell_1$), $g(\\zeta) = \\zeta$", "DyLam-Normalizer-l1"),
    Arm("Min--max", "DyLam-Normalizer-minmax-fixed"),
    Arm("Min--max, reversed routing (control)", "DyLam-Normalizer-minmax"),
]

ENV = "CHICKENBANANA"
REWARD_METRICS = {"Banana": "ep_info/Banana", "Chicken": "ep_info/Chicken",
                   "Gate": "ep_info/Objective"}  # Gate's wandb key, see trad/methods.py
LAMBDA_METRICS = {"Banana": "lambdas/Banana", "Chicken": "lambdas/Chicken",
                   "Gate": "lambdas/Objective"}
COMPONENTS = ["Banana", "Chicken", "Gate"]
