#!/usr/bin/env bash
# Rerun every Chicken-Banana arm the paper shows, on the 0/1 rewards of 99e13c5.
#
#   phase A  Baseline Decq Udc Dylam Dylam_Scalar Dylam_Lagrangian (+ eta sweep)
#   phase B  extract DyLam's new mean lambda(t), then replay it as Dylam_Openloop
#
# Seeds 1..SEEDS, explicit. ~4 s per run, so runs go one at a time. A finished
# run drops a .done marker and is skipped on re-run. Phase B needs wandb (TRACK=true),
# since the schedule is read from the phase-A DyLam runs.
#
#   ./run_cb.sh              # both phases
#   ./run_cb.sh a            # phase A only
#   SEEDS=1 TRACK=false ./run_cb.sh a
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-10}
TRACK=${TRACK:-true}
LOGDIR=${LOGDIR:-logs/cb}
PY=${PY:-../.venv/bin/python}
phase=${1:-all}
mkdir -p "$LOGDIR"

run() {  # setup env seed
  local tag="$2-$1-$3"
  if [[ -f "$LOGDIR/$tag.done" ]]; then echo "skip $tag"; return; fi
  "$PY" train_q_learning.py --setup "$1" --env "$2" --seed "$3" --track "$TRACK" \
    > "$LOGDIR/$tag.log" 2>&1 || { echo "FAILED $tag: see $LOGDIR/$tag.log" >&2; exit 1; }
  touch "$LOGDIR/$tag.done"; echo "done $tag"
}

if [[ "$phase" == all || "$phase" == a ]]; then
  for seed in $(seq 1 "$SEEDS"); do
    for setup in Baseline Decq Udc Dylam Dylam_Scalar Dylam_Lagrangian; do
      run "$setup" CHICKENBANANA "$seed"
    done
    run Dylam_Lagrangian CHICKENBANANA_ETA01 "$seed"
    run Dylam_Lagrangian CHICKENBANANA_ETA10 "$seed"
  done
fi

if [[ "$phase" == all || "$phase" == b ]]; then
  "$PY" ../result_analysis/extract_lambda_schedule.py --env CHICKENBANANA \
    --components Gate Banana Chicken --out schedules/chickenbanana_dylam_lambda.csv
  for seed in $(seq 1 "$SEEDS"); do run Dylam_Openloop CHICKENBANANA "$seed"; done
fi
echo "all CB runs finished"
