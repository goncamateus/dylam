"""Canonical metadata for every (paper column, method) cell in the trad scope.

fetch_data.py, table.py, and figure.py all read this one place. Column here
is the *paper* environment/column identity (e.g. "HalfCheetah-v4 (env
return)" is its own column though it shares runs with "HalfCheetah-v4"), not
the raw wandb env tag, which can differ per method within one column:
Tuned-UDC's runs live under the wandb env VSS_TUNED, not VSS, even though it
is presented as just another VSS-v0 method.
"""
from collections import namedtuple

Cell = namedtuple("Cell", "column method wandb_env wandb_setup metric")

# Method -> wandb setup, per environment. The source of truth for both the
# main table cells and the Chicken--Banana per-component figure cells below,
# so the four methods and their setups are declared exactly once.
CHICKENBANANA_METHODS = {
    "Base SO RL": "Baseline", "Q-Decomposition": "Decq", "UDC": "Drq", "DyLam": "Dylam",
}
HALFCHEETAH_METHODS = {"Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}
VSS_METHODS = {"Base SO RL": "Baseline", "UDC": "Drq", "DyLam": "Dylam"}

CELLS = [
    *(Cell("Chicken--Banana", m, "CHICKENBANANA", s, "ep_info/total")
      for m, s in CHICKENBANANA_METHODS.items()),
    # The ablation of Section 6.1.1 (DyLam's weights, no Q-decomposition): a
    # row of tab:res/trad/summary and tab:res/trad/iqm, not a rival method.
    Cell("Chicken--Banana", "DyLam-Scalar", "CHICKENBANANA", "Dylam_Scalar", "ep_info/total"),

    *(Cell("HalfCheetah-v4", m, "HALFCHEETAH", s, "ep_info/Final_position")
      for m, s in HALFCHEETAH_METHODS.items()),
    # Same runs, re-scored on the environment's own scalar reward.
    *(Cell("HalfCheetah-v4 (env return)", m, "HALFCHEETAH", s, "ep_info/total")
      for m, s in HALFCHEETAH_METHODS.items()),

    *(Cell("VSS-v0", m, "VSS", s, "ep_info/Goal") for m, s in VSS_METHODS.items()),
    Cell("VSS-v0", "Tuned-UDC", "VSS_TUNED", "Drq", "ep_info/Goal"),
]

# Chicken--Banana per-component curves (fig:res/chicken_banana/components):
# the same four methods as the main Chicken--Banana column, three more metrics.
# Gate's wandb key is "Objective", not "Gate" -- a pre-existing mismatch
# between comp_names/the manuscript's name for this component and what
# src/dylam/envs/chicken_banana.py actually logs it as (training code, out
# of scope for this analysis migration to fix).
COMPONENT_METRICS = {
    "Banana": "ep_info/Banana", "Chicken": "ep_info/Chicken", "Gate": "ep_info/Objective",
}
COMPONENT_CELLS = [
    Cell(f"Chicken--Banana {comp}", m, "CHICKENBANANA", s, metric)
    for comp, metric in COMPONENT_METRICS.items()
    for m, s in CHICKENBANANA_METHODS.items()
]
