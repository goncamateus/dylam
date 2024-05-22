import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from methods.networks.raw.continuous import MLPPolicy, QNetwork
from methods.networks.targets import TargetCritic

from utils.buffer import ReplayBuffer, StratLastRewards
from utils.ou_noise import OrnsteinUhlenbeckNoise as OUNoise


class DDPG(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        hidden_dim=256,
    ):
        super(DDPG, self).__init__()

        # General Hyperparameters
        self.args = args
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = args.gamma
        self.lr_actor = args.policy_lr
        self.lr_critic = args.q_lr
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.num_inputs = np.array(observation_space.shape).prod()
        self.num_actions = np.array(action_space.shape).prod()
        self.noises = [
            OUNoise(mu=np.zeros(action_space.shape), sigma=0.2)
        ] * args.num_envs
        [noise.reset() for noise in self.noises]
        self.update_counter = 0
        self.actor, self.critic = self.get_networks(hidden_dim)
        self.critic_target = TargetCritic(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)
        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def get_networks(self, hidden_dim):
        actor = MLPPolicy(self.num_inputs, self.num_actions, hidden_dim=hidden_dim)
        critic = QNetwork(
            self.num_inputs,
            self.num_actions,
            hidden_dim=hidden_dim,
        )
        return actor, critic

    def get_replay_buffer(self, buffer_size):
        return ReplayBuffer(buffer_size, self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.target_model.to(device)
        return super(DDPG, self).to(device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        action = self.actor.get_action(state)
        for i, noise in enumerate(self.noises):
            action[i] = noise(action[i])
        return action

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        with torch.no_grad():
            next_state_action = self.actor(next_state_batch)
            qf_next_target = self.critic_target(next_state_batch, next_state_action)
            qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * qf_next_target

        qf = self.critic(state_batch, action_batch)
        qf_loss = F.mse_loss(qf, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        return qf_loss

    def update_actor(self, state_batch):
        pi = self.actor(state_batch)
        qf_pi = self.critic(state_batch, pi)

        # Jπ = 𝔼st∼D,εt∼N[− Q(st,f(εt;st))]
        policy_loss = -qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        self.update_counter += 1
        if self.update_counter % 1500 == 0:
            for i in range(len(self.noises)):
                self.noises[i].sigma *= 0.99
            self.update_counter = 0
        return policy_loss

    def update(self, batch_size, update_actor=False):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size, continuous=True)

        reward_batch = reward_batch * self.reward_scaling
        qf_loss = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        policy_loss = None
        if update_actor:
            policy_loss = self.update_actor(state_batch)

        return policy_loss, qf_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.critic.load_state_dict(torch.load(path + "critic.pt"))
        self.actor.eval()
        self.critic.eval()
