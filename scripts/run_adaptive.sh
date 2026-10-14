#!/usr/bin/env bash
# R5 adaptive baseline (primal-dual only; the linear schedule is not run), 10 seeds each, one environment at a time:
#
#   CHICKENBANANA  Dylam_Lagrangian                  10 runs in parallel
#   HALFCHEETAH    Dylam_Lagrangian                   7 runs in parallel
#   VSS            Dylam_Lagrangian                   5 runs in parallel
#
# Seeds 1..SEEDS, explicit. A finished run drops a .done marker (same tag format as
# run_r5.sh / run_cb.sh, LOGDIR=logs/r5) and is skipped on re-run.
#
#   ./run_adaptive.sh               # cb, then hc, then vss
#   ./run_adaptive.sh hc            # one env only: cb | hc | vss
#   SEEDS=2 TRACK=false ./run_adaptive.sh cb --dry-run
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=${SEEDS:-10}
TRACK=${TRACK:-true}
LOGDIR=${LOGDIR:-logs/r5}
PY=${PY:-../.venv/bin/python}
export LOGDIR TRACK PY

which_set=${1:-all}
dry=""
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && dry="echo [dry-run]"
[[ "$which_set" == "--dry-run" ]] && which_set=all
mkdir -p "$LOGDIR"

phase() {  # parallel script env setup...
  local parallel=$1 script=$2 env=$3; shift 3
  local jobs=() i setup
  for i in $(seq 1 "$SEEDS"); do
    for setup in "$@"; do jobs+=("$script $setup $env $i"); done
  done
  echo "== $env: ${#jobs[@]} runs, $parallel at a time =="
  printf '%s\n' "${jobs[@]}" | $dry xargs -P "$parallel" -I{} bash -c '
    read -r script setup env seed <<< "{}"
    tag="${env}-${setup}-${seed}"
    if [[ -f "$LOGDIR/${tag}.done" ]]; then echo "skip ${tag}"; exit 0; fi
    echo "start ${tag}"
    if "$PY" "$script" --setup "$setup" --env "$env" --seed "$seed" --track "$TRACK" \
         > "$LOGDIR/${tag}.log" 2>&1; then
      touch "$LOGDIR/${tag}.done"; echo "done  ${tag}"
    else
      echo "FAILED ${tag}: see $LOGDIR/${tag}.log" >&2; exit 1
    fi
  '
}

case "$which_set" in all|cb|hc|vss) ;; *) echo "unknown set '$which_set' (use: all | cb | hc | vss)" >&2; exit 2 ;; esac
[[ "$which_set" == all || "$which_set" == cb  ]] && phase 10 train_q_learning.py CHICKENBANANA Dylam_Lagrangian
[[ "$which_set" == all || "$which_set" == hc  ]] && phase 7  train.py HALFCHEETAH Dylam_Lagrangian
[[ "$which_set" == all || "$which_set" == vss ]] && phase 5  train.py VSS Dylam_Lagrangian
echo "all adaptive runs finished"
