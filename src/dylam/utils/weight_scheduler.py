import math

import numpy as np


def generate_weights(count: int, n: int, m: int) -> np.ndarray:
    if m == 1:
        weights = np.random.dirichlet([1] * n, size=count)
    else:
        weights = np.random.dirichlet([1 / m] * n, size=count)
    return weights.astype(np.float32)


class WeightSchedule:
    def __init__(
        self,
        total_steps: int,
        num_rewards: int,
        weight_change_freq: int = 1000,
        mode: str = "regular",
        seed: int = 0,
    ):
        self.weight_change_freq = weight_change_freq
        rng_state = np.random.get_state()
        np.random.seed(seed)
        count = math.ceil(total_steps / weight_change_freq) + 1
        m = 1 if mode == "sparse" else 10
        self.weights = generate_weights(count, num_rewards, m)
        np.random.set_state(rng_state)

    def current_w(self, step: int) -> np.ndarray:
        idx = min(step // self.weight_change_freq, len(self.weights) - 1)
        return self.weights[idx]

    def changed(self, step: int) -> bool:
        return step > 0 and step % self.weight_change_freq == 0

    def __len__(self) -> int:
        return len(self.weights)
