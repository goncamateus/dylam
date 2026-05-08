import random

import numpy as np
import torch

from dylam.utils.buffer import ReplayWeightAwareBuffer


def _crowding_distances(weights: np.ndarray) -> np.ndarray:
    n = len(weights)
    if n <= 2:
        return np.full(n, np.inf)

    dims = weights.shape[1]
    distances = np.zeros(n)

    for d in range(dims):
        order = np.argsort(weights[:, d])
        sorted_w = weights[order, d]
        w_range = sorted_w[-1] - sorted_w[0]

        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        if w_range == 0:
            continue

        for i in range(1, n - 1):
            distances[order[i]] += (sorted_w[i + 1] - sorted_w[i - 1]) / w_range

    return distances


class DiverseReplayBuffer(ReplayWeightAwareBuffer):
    def __init__(self, max_size, device="cpu", secondary_size=10000):
        super().__init__(max_size, device)
        self.secondary_size = secondary_size
        self.secondary = []

    def _try_promote(self, experience):
        candidate_w = experience[5]

        if len(self.secondary) < self.secondary_size:
            self.secondary.append(experience)
            return

        sec_weights = np.array([e[5] for e in self.secondary])
        current_distances = _crowding_distances(sec_weights)

        candidate_weights = np.vstack([sec_weights, candidate_w])
        new_distances = _crowding_distances(candidate_weights)

        # candidate increases diversity if its own distance in the extended set
        # is greater than the minimum existing distance it would replace
        min_idx = np.argmin(current_distances[np.isfinite(current_distances)] if np.any(np.isfinite(current_distances)) else current_distances)
        finite_mask = np.isfinite(current_distances)
        if not np.any(finite_mask):
            return
        evict_idx = np.where(finite_mask)[0][np.argmin(current_distances[finite_mask])]

        candidate_dist = new_distances[-1]
        if candidate_dist > current_distances[evict_idx]:
            self.secondary[evict_idx] = experience

    def add(self, state, action, reward, next_state, done, weights):
        for i in range(len(state)):
            rew = reward[i]
            act = action[i]
            weight = weights[i]
            if rew.shape == ():
                rew = np.array([rew])
            if act.shape == ():
                act = np.array([act])
            if weight.shape == ():
                weight = np.array([weight])
            experience = (
                state[i],
                act,
                rew,
                next_state[i],
                done[i],
                weight,
            )
            if len(self.buffer) >= self.max_size:
                evicted = self.buffer[self.ptr]
                self._try_promote(evicted)
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
            self.ptr = int((self.ptr + 1) % self.max_size)

    def _pack_batch(self, batch):
        states, actions, rewards, next_states, dones, weights = map(
            np.array, zip(*batch)
        )
        states_v = torch.Tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_v = torch.Tensor(rewards).to(self.device)
        last_states_v = torch.Tensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        weights_v = torch.Tensor(weights).to(self.device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t, weights_v

    def sample(self, batch_size):
        min_secondary = 2
        if len(self.secondary) >= min_secondary:
            n_sec = batch_size // 2
            n_rec = batch_size - n_sec
            n_rec = min(n_rec, len(self.buffer))
            n_sec = min(n_sec, len(self.secondary))
            batch = random.sample(self.buffer, n_rec) + random.sample(self.secondary, n_sec)
        else:
            batch = random.sample(self.buffer, batch_size)
        return self._pack_batch(batch)
