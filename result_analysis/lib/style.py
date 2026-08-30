"""Shared plot style: method/condition colours, per-seed smoothing.

Populated scope by scope as each generator migrates, with only what the
migrated scopes actually use -- notebooks and scripts not yet migrated still
carry their own (already-drifted) copies of tick formatting, fonts, etc.
until they are.
"""
import pandas as pd

METHOD_COLORS = {
    "Base SO RL": "tab:orange",
    "Q-Decomposition": "tab:purple",
    "UDC": "tab:blue",
    "DyLam": "tab:green",
    # Deliberately shared with Q-Decomposition: the two never appear in the
    # same axes, and fig:res/all's caption documents this pairing by name --
    # changing it would make authored prose wrong.
    "Tuned-UDC": "tab:purple",
    # Deliberately shared with Base SO RL: never in the same axes (morl's
    # figures don't plot trad's single-objective baseline).
    "GPI-LS": "tab:orange",
    "PGMORL": "tab:blue",
    # Deliberately shared with PGMORL: never in the same axes (PGMORL is
    # HalfCheetah-only, DynMORL is Minecart-only).
    "DynMORL": "tab:blue",
}

CONDITION_COLORS = {
    "Nominal": "dimgray",
    "Move $-25\\%$": "tab:blue",
    "Move $+50\\%$": "tab:red",
    "Ball $+25\\%$": "tab:purple",
    "Ball $-25\\%$": "tab:brown",
    "Move $+50\\%$, ball $+25\\%$": "tab:orange",
    "Move $-25\\%$, ball $-50\\%$": "tab:green",
}


def rolling_smooth(value, window=None):
    """Per-seed rolling mean, applied before cross-seed aggregation.

    Aggregating raw per-step values is noisy enough at the episodic sampling
    rate that the aggregate curve hides the trend the summary statistics
    report; smoothing each seed's series first, on its own step grid, is what
    the published figure does.
    """
    if window is None:
        window = max(5, len(value) // 40)
    return pd.Series(value).rolling(window, min_periods=1).mean().to_numpy(dtype=float)
