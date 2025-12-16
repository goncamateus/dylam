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


class ReplayWeightAwareBuffer(ReplayBuffer):
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
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.ptr] = experience
            self.ptr = int((self.ptr + 1) % self.max_size)

    def sample(self, batch_size):
        """From a batch of experience, return values in Tensor form on device"""
        batch = random.sample(self.buffer, batch_size)
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


class PPOBuffer:
    def __init__(
        self,
        state_dim,
        action_dim,
        num_rewards,
        num_envs,
        num_steps,
        batch_size,
        minibatch_size,
        reward_scaling,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_rewards = num_rewards
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.reward_scaling = reward_scaling
        self.device = device
        self.ptr = 0
        self.clear()
        self.to(device)

    def to(self, device):
        self.device = device
        self.ptr = 0
        self._observations = self._observations.to(device)
        self._actions = self._actions.to(device)
        self._rewards = self._rewards.to(device)
        self._dones = self._dones.to(device)
        self._log_probs = self._log_probs.to(device)
        self._values = self._values.to(device)
        self._returns = self._returns.to(device)
        self._advantages = self._advantages.to(device)
        return self

    def clear(self):
        self._observations = torch.zeros(
            size=(self.num_steps, self.num_envs, self.state_dim)
        ).float()
        self._actions = torch.zeros(
            size=(self.num_steps, self.num_envs, self.action_dim)
        ).long()
        self._rewards = torch.zeros(
            size=(self.num_steps, self.num_envs, self.num_rewards)
        ).float()
        self._dones = torch.zeros(size=(self.num_steps, self.num_envs)).bool()
        self._log_probs = torch.zeros(size=(self.num_steps, self.num_envs)).float()
        self._values = torch.zeros(
            size=(self.num_steps, self.num_envs, self.num_rewards)
        ).float()
        self._returns = torch.zeros(
            size=(self.num_steps, self.num_envs, self.num_rewards)
        ).float()
        self._advantages = torch.zeros(
            size=(self.num_steps, self.num_envs, self.num_rewards)
        ).float()

    def add(self, state, action, reward, done, log_prob, value):
        current = self.ptr % self.num_steps
        self._observations[current] = torch.tensor(state).float().to(self.device)
        self._actions[current] = torch.tensor(action).long().to(self.device)
        self._rewards[current] = (
            torch.tensor(reward).float().to(self.device) * self.reward_scaling
        )
        self._dones[current] = torch.tensor(done).bool().to(self.device)
        self._log_probs[current] = log_prob
        self._values[current] = value
        self.ptr += 1

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return torch.Tensor(self._rewards).to(self.device)

    @property
    def dones(self):
        return torch.Tensor(self._dones).to(self.device)

    @property
    def log_probs(self):
        return self._log_probs

    @property
    def values(self):
        return self._values.to(self.device)

    @property
    def returns(self):
        return self._returns

    @property
    def advantages(self):
        return self._advantages

    @returns.setter
    def returns(self, returns):
        self._returns = returns

    @advantages.setter
    def advantages(self, advantages):
        self._advantages = advantages

    def sample(self):
        obs = self.observations.reshape(-1, self.state_dim).to(self.device)
        acts = self.actions.reshape(-1, 1).to(self.device)
        logp = self.log_probs.reshape(-1).to(self.device)
        adv = self.advantages.reshape(-1, self.num_rewards).to(self.device)
        ret = self.returns.reshape(-1, self.num_rewards).to(self.device)
        val = self.values.reshape(-1, self.num_rewards).to(self.device)
        for step_idx in range(0, self.batch_size, self.minibatch_size):
            observations = obs[step_idx : step_idx + self.minibatch_size]
            log_probs = logp[step_idx : step_idx + self.minibatch_size]
            actions = acts[step_idx : step_idx + self.minibatch_size]
            advantages = adv[step_idx : step_idx + self.minibatch_size]
            returns = ret[step_idx : step_idx + self.minibatch_size]
            values = val[step_idx : step_idx + self.minibatch_size]
            batch = (observations, log_probs, actions, advantages, returns, values)
            yield batch

    def __len__(self):
        return len(self._observations)
