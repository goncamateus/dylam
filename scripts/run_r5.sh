#!/usr/bin/env bash
# Launch the R5 adaptive-baseline runs (ticket 10, ADR 0003), locally.
#
#   Dylam_Lagrangian  HC, VSS   (CB, with its eta sweep, lives in run_cb.sh)
#   Dylam_Linear      HC, VSS   (CB omitted: native scalarization = uniform = UDC)
#
# 10 seeds each. A finished run drops a .done marker and is skipped on re-run.
# Seeds: with SEED_BASE unset no --seed is passed (seed 0, as slurm_job.sh ran the
# DyLam rows); set SEED_BASE=37000 for explicit seeds 37001..37010 instead.
#
#   ./run_r5.sh                     # everything, 5 heavy runs at a time
#   ./run_r5.sh hc                  # one set only: hc | vss
#   SEEDS=2 TRACK=false ./run_r5.sh hc --dry-run
set -euo pipefail
cd "$(dirname "$0")"

PARALLEL=${PARALLEL:-5}
SEEDS=${SEEDS:-10}
TRACK=${TRACK:-true}
LOGDIR=${LOGDIR:-logs/r5}
PY=${PY:-../.venv/bin/python}
SEED_BASE=${SEED_BASE:-}

which_set=${1:-all}
dry=""
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && dry="echo [dry-run]"
[[ "${which_set}" == "--dry-run" ]] && which_set=all
mkdir -p "$LOGDIR"

jobs=()
for i in $(seq 1 "$SEEDS"); do
  seed=${SEED_BASE:+$((SEED_BASE + i))}
  seed=${seed:-none}
  case "$which_set" in
    all|hc)  jobs+=("train.py Dylam_Lagrangian HALFCHEETAH $seed $i"
                    "train.py Dylam_Linear HALFCHEETAH $seed $i") ;;&
    all|vss) jobs+=("train.py Dylam_Lagrangian VSS $seed $i"
                    "train.py Dylam_Linear VSS $seed $i") ;;
    hc) ;;
    *) echo "unknown set '$which_set' (use: all | hc | vss)" >&2; exit 2 ;;
  esac
done

echo "${#jobs[@]} runs, $PARALLEL at a time, logs in $LOGDIR"
printf '%s\n' "${jobs[@]}" | $dry xargs -P "$PARALLEL" -I{} bash -c '
  set -euo pipefail
  read -r script setup env seed i <<< "{}"
  tag="${env}-${setup}-${i}"
  seed_flag=""
  [[ "$seed" != none ]] && seed_flag="--seed $seed"
  if [[ -f "'"$LOGDIR"'/${tag}.done" ]]; then echo "skip ${tag} (already done)"; exit 0; fi
  echo "start ${tag}"
  if '"$PY"' "$script" --setup "$setup" --env "$env" $seed_flag \
       --track '"$TRACK"' > "'"$LOGDIR"'/${tag}.log" 2>&1; then
    touch "'"$LOGDIR"'/${tag}.done"; echo "done  ${tag}"
  else
    echo "FAILED ${tag}: see '"$LOGDIR"'/${tag}.log" >&2; exit 1
  fi
'
echo "all launched runs finished"
