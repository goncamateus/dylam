"""Launch the two Chicken-Banana ablations the R1 revision needs, 10 seeds each.

  1. Dylam_Scalar            DyLam weights, single Q-table (no Q-decomposition)
  2. DyLam-Normalizer-minmax-fixed   min-max transform with the routing direction
                             corrected (the shipped `minmax` reverses it)
"""
import time

from dylam.utils.experiment import get_experiment, parse_args, setup_run
from dylam.utils.logger import QLogger
from train_q_learning import train

SEEDS = 10


def run(params, setup, exp_prefix):
    params.setup = setup
    for i in range(SEEDS):
        params.seed = int(time.time()) + i
        exp_name = f"{exp_prefix}_{params.seed}"
        print(f"=== {setup} seed {i + 1}/{SEEDS} ===", flush=True)
        logger = QLogger(exp_name, params)
        setup_run(params)
        train(params, exp_name, logger)
        logger.close()


if __name__ == "__main__":
    args = parse_args()
    params = get_experiment(args)
    if params.scalar_critic:
        run(params, "Dylam_Scalar", "ChickenBanana-Scalar")
    else:
        params.normalizer = "minmax-fixed"
        run(params, "DyLam-Normalizer-minmax-fixed", "ChickenBanana-Normalizer-minmax-fixed")
