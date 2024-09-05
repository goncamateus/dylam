import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, device="cpu"):
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        for i in range(len(state)):
            rew = reward[i]
            act = action[i]
            if rew.shape == ():
                rew = np.array([rew])
            if act.shape == ():
                act = np.array([act])
            experience = (
                state[i],
                act,
                rew,
                next_state[i],
                done[i],
            )
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
            self.ptr = int((self.ptr + 1) % self.max_size)

    def clear(self):
        self.buffer.clear()
        self.ptr = 0

    def sample(self, batch_size):
        """From a batch of experience, return values in Tensor form on device"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states_v = torch.Tensor(states).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_v = torch.Tensor(rewards).to(self.device)
        last_states_v = torch.Tensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        return states_v, actions_v, rewards_v, last_states_v, dones_t

    def __len__(self):
        return len(self.buffer)


class StratLastRewards:
    def __init__(self, size, n_rewards):
        self.pos = 0
        self.size = size
        self._can_do = False
        self.rewards = np.zeros((size, n_rewards))

    def add(self, reward):
        self.rewards[self.pos] = reward
        if self.pos == (self.size - 1):
            self._can_do = True
        self.pos = (self.pos + 1) % self.rewards.shape[0]

    def can_do(self):
        return self._can_do

    def mean(self):
        return self.rewards.mean(0)
