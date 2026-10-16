#!/usr/bin/env bash
# Rerun every Chicken-Banana arm the paper shows, on the 0/1 rewards of 99e13c5.
#
#   phase A  ARMS (default: Baseline Decq Udc Dylam Dylam_Scalar Dylam_Lagrangian + eta sweep)
#   phase B  extract DyLam's mean lambda(t) over its newest SEEDS runs, then replay it
#            as Dylam_Openloop
#
# Seeds 1..SEEDS, explicit, JOBS runs at a time. A finished run drops a .done marker
# and is skipped on re-run. Phase B needs wandb (TRACK=true), since the schedule is read
# from the phase-A DyLam runs; a new schedule invalidates finished Dylam_Openloop runs,
# so delete their .done markers before re-running phase B.
#
#   ./run_cb.sh              # both phases
#   ./run_cb.sh a            # phase A only
#   SEEDS=1 TRACK=false ./run_cb.sh a
#   SEEDS=100 ARMS="Baseline Decq Udc Dylam Dylam_Scalar" ./run_cb.sh
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-10}
JOBS=${JOBS:-10}
TRACK=${TRACK:-true}
LOGDIR=${LOGDIR:-logs/cb}
PY=${PY:-../.venv/bin/python}
# setup[:env], env defaulting to CHICKENBANANA
ARMS=${ARMS:-"Baseline Decq Udc Dylam Dylam_Scalar Dylam_Lagrangian
  Dylam_Lagrangian:CHICKENBANANA_ETA01 Dylam_Lagrangian:CHICKENBANANA_ETA10"}
phase=${1:-all}
mkdir -p "$LOGDIR"

run() {  # setup env seed
  local tag="$2-$1-$3"
  if [[ -f "$LOGDIR/$tag.done" ]]; then echo "skip $tag"; return; fi
  "$PY" train_q_learning.py --setup "$1" --env "$2" --seed "$3" --track "$TRACK" \
    > "$LOGDIR/$tag.log" 2>&1 || { echo "FAILED $tag: see $LOGDIR/$tag.log" >&2; exit 1; }
  touch "$LOGDIR/$tag.done"; echo "done $tag"
}

spawn() {  # run in the background, at most JOBS at a time
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  run "$@" &
}

if [[ "$phase" == all || "$phase" == a ]]; then
  for seed in $(seq 1 "$SEEDS"); do
    for arm in $ARMS; do
      env=CHICKENBANANA; [[ "$arm" == *:* ]] && env=${arm#*:}
      spawn "${arm%%:*}" "$env" "$seed"
    done
  done
  wait
fi

if [[ "$phase" == all || "$phase" == b ]]; then
  "$PY" ../result_analysis/extract_lambda_schedule.py --env CHICKENBANANA \
    --components Gate Banana Chicken --max-seeds "$SEEDS" \
    --out schedules/chickenbanana_dylam_lambda.csv
  for seed in $(seq 1 "$SEEDS"); do spawn Dylam_Openloop CHICKENBANANA "$seed"; done
  wait
fi
echo "all CB runs finished"
