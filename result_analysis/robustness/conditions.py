"""Canonical metadata for the seven RQ3 robustness conditions.

fetch_data.py, table.py, and figure.py all read this one place, so adding an
eighth condition is one edit here, not one in each of the three. Draw/row
order is each consumer's own concern (the table groups rows by which ceiling
moved; the figure keeps the legend order of the already-published plot) and
is not repeated here.
"""
from collections import namedtuple

Condition = namedtuple("Condition", "label env setup r_max section")

CONDITIONS = {
    "nominal": Condition(
        "Nominal", "VSS", "Dylam", "(150, 40, -100)", None),
    "move_m25": Condition(
        "Move $-25\\%$", "ROBUSTNESS_MOVE2", "Dylam", "(112.5, 40, -100)", "Move ceiling"),
    "move_p50": Condition(
        "Move $+50\\%$", "ROBUSTNESS_MOVE1", "Dylam", "(225, 40, -100)", None),
    "ball_m25": Condition(
        "Ball $-25\\%$", "ROBUSTNESS_BALL_M25", "Dylam", "(150, 30, -100)", "Ball-to-goal ceiling"),
    "ball_p25": Condition(
        "Ball $+25\\%$", "ROBUSTNESS_BALL_P25", "Dylam", "(150, 50, -100)", None),
    "compound_move_p50_ball_p25": Condition(
        "Move $+50\\%$, ball $+25\\%$", "ROBUSTNESS_BALL1", "Dylam", "(225, 50, -100)", "Compound"),
    "compound_move_m25_ball_m50": Condition(
        "Move $-25\\%$, ball $-50\\%$", "ROBUSTNESS_BALL2", "Dylam", "(112.5, 20, -100)", None),
}
