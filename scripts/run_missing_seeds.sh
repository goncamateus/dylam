#!/bin/bash
# Top up the learning-dynamics baselines to K = 10 finished seeds each.
#
# Current finished-run counts on goncamateus/DyLam (checked 2026-08-26):
#   HALFCHEETAH Baseline 5, Drq 6 | VSS Baseline 5, Drq 5 | VSS_TUNED Drq 7
# DyLam itself already has 10 seeds in every environment.
#
# It also launches the three RQ3 robustness conditions that were never run in the
# intended one-at-a-time +/-25% design. What actually ran (per the wandb run
# configs) was: MOVE1 = move +50%; MOVE2 = move -25%; BALL1 = move +50% AND ball
# +25%; BALL2 = move -25% AND ball -50%. Only MOVE2 matches the paper's stated
# design, so MOVE_P25, BALL_P25 and BALL_M25 (added to experiments.yml) supply
# the missing three. Nominal r_max = (150, 40, -100).
#
# Usage:  cd scripts && ./run_missing_seeds.sh [--dry-run]

set -euo pipefail
DRY=${1:-}

# env  setup  n_missing
JOBS=(
  "VSS         Baseline 5"
  "VSS         Udc      5"
  "VSS_TUNED   Udc      3"
)

for job in "${JOBS[@]}"; do
    read -r ENV SETUP N <<< "$job"
    CMD="sbatch --job-name=dylam-${ENV,,}-${SETUP,,} slurm_job_n.sh train.py $ENV $SETUP $N"
    if [[ "$DRY" == "--dry-run" ]]; then
        echo "$CMD"
    else
        echo "+ $CMD"
        eval "$CMD"
    fi
done

echo
echo "52 runs queued (22 baseline top-up + 30 robustness)."
echo "When they finish, refresh the tables with:"
echo "  python ../result_analysis/table1_update.py"
echo "  python ../result_analysis/robustness_update.py"
