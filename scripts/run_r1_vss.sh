#!/usr/bin/env bash
# Launch the two VSS-v0 run sets the R1 revision still needs, locally.
#
#   ROBUSTNESS_MOVE_P25   move ceiling +25%, the missing one-at-a-time RQ3 condition
#                         (the earlier attempt left no finished runs)
#   Dylam_Openloop        R5: replay of DyLam's own mean lambda(t) with no return
#                         feedback and no bounds
#
# Each is 10 seeds of 500k steps. VSS training is heavy, so at most PARALLEL run
# concurrently. A finished seed drops a .done marker and is skipped on re-run, so
# the script can be interrupted and restarted.
#
#   ./run_r1_vss.sh                 # both sets, 5 at a time
#   PARALLEL=3 ./run_r1_vss.sh      # fewer concurrent runs
#   SEEDS=2 TRACK=false ./run_r1_vss.sh --dry-run
#   ./run_r1_vss.sh openloop        # one set only: openloop | movep25
set -euo pipefail
cd "$(dirname "$0")"

PARALLEL=${PARALLEL:-5}
SEEDS=${SEEDS:-10}
TRACK=${TRACK:-true}
LOGDIR=${LOGDIR:-logs/r1}
PY=${PY:-../.venv/bin/python}

which_set=${1:-all}
dry=""
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && dry="echo [dry-run]"
[[ "${which_set}" == "--dry-run" ]] && which_set=all

mkdir -p "$LOGDIR"

if [[ ! -f schedules/vss_dylam_lambda.csv ]]; then
  echo "missing schedules/vss_dylam_lambda.csv" >&2
  echo "generate it first: python ../result_analysis/extract_lambda_schedule.py" >&2
  exit 1
fi

jobs=()
for i in $(seq 1 "$SEEDS"); do
  case "$which_set" in
    all|movep25)  jobs+=("dylam robustness_move_p25 $((17000 + i))") ;;&
    all|openloop) jobs+=("dylam_openloop vss $((27000 + i))") ;;
    *) echo "unknown set '$which_set' (use: all | movep25 | openloop)" >&2; exit 2 ;;
  esac
done

echo "${#jobs[@]} runs, $PARALLEL at a time, logs in $LOGDIR"
printf '%s\n' "${jobs[@]}" | $dry xargs -P "$PARALLEL" -I{} bash -c '
  set -euo pipefail
  read -r setup env seed <<< "{}"
  tag="${env}-${setup}-${seed}"
  if [[ -f "'"$LOGDIR"'/${tag}.done" ]]; then
    echo "skip ${tag} (already done)"
    exit 0
  fi
  echo "start ${tag}"
  if '"$PY"' train.py --setup "$setup" --env "$env" --seed "$seed" \
       --track '"$TRACK"' > "'"$LOGDIR"'/${tag}.log" 2>&1; then
    touch "'"$LOGDIR"'/${tag}.done"
    echo "done  ${tag}"
  else
    echo "FAILED ${tag}: see '"$LOGDIR"'/${tag}.log" >&2
    exit 1
  fi
'
echo "all launched runs finished"
