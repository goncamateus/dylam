"""Canonical per-environment component/weight registry for fig:curr/weights
and fig:curr/components (sections/results/trad/curriculum.tex, app.tex).

Curriculum has one method (DyLam) and no rival comparison: it is about one
policy's own adaptive lambda trajectory and the per-component returns
driving it, not a method comparison the way trad/robustness are.

`name` is each component's actual wandb identity: a run's own logged
`comp_names` config, which is what both `ep_info/<name>` and
`lambdas/<name>` are keyed by (src/dylam/utils/logger.py). Confirmed
empirically per env, since scripts/experiments.yml's *current* comp_names
isn't guaranteed to match what already-finished runs logged: HalfCheetah
and VSS runs match today's YAML (['Run','Control'] and
['Move','Ball','Energy']), but ChickenBanana runs log "Objective" for
their first component, not the YAML's current "Gate" --
`lambdas/Objective` fetches real data, `lambdas/Gate` fetches zero rows.

`label` is the manuscript's own display name for the component
(curriculum.tex), used only for the legend -- "Gate" for ChickenBanana's
"Objective", identical to `name` everywhere else.

`ep_metric` is `reward_<name>` with the `reward_` prefix stripped to
`ep_info/` (log_episode) -- and each env wrapper's own
cumulative_reward_info dict (src/dylam/envs/) doesn't always spell that
the same as `name`: HalfCheetah logs the legacy `reward_run`/`reward_ctrl`
(lowercase, abbreviated) regardless of comp_names' "Run"/"Control", so
its `ep_metric` diverges from `name` where ChickenBanana/VSS's don't.
"""
from collections import namedtuple

Component = namedtuple("Component", "name label ep_metric r_max r_min")
Env = namedtuple("Env", "gym_label setup xlabel components")

ENVS = {
    "CHICKENBANANA": Env("ChickenBanana-v0", "Dylam", "Episode", [
        Component("Objective", "Gate", "ep_info/Objective", 100, 0),
        Component("Banana", "Banana", "ep_info/Banana", 30, 0),
        Component("Chicken", "Chicken", "ep_info/Chicken", 70, 0),
    ]),
    "HALFCHEETAH": Env("HalfCheetah-v4", "Dylam", "Environment step", [
        Component("Run", "Run", "ep_info/run", 800, 0),
        Component("Control", "Control", "ep_info/ctrl", -200, -800),
    ]),
    "VSS": Env("VSS-v0", "Dylam", "Environment step", [
        Component("Move", "Move", "ep_info/Move", 150, 0),
        Component("Ball", "Ball", "ep_info/Ball", 40, 0),
        Component("Energy", "Energy", "ep_info/Energy", -100, -300),
    ]),
}
MAX_SEEDS = 10
