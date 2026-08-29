"""Shared plot style: condition colours, tick formatting, per-seed smoothing.

Populated scope by scope as each generator migrates. Notebooks and scripts
not yet migrated still carry their own (already-drifted) copies of some of
this until they are.
"""
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import font_manager

FORMATTER = ticker.ScalarFormatter(useMathText=True)
FORMATTER.set_scientific(True)
FORMATTER.set_powerlimits((-1, 1))
FONT = font_manager.FontProperties(weight="bold")

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
