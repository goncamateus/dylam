import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from methods.networks.raw.continuous import MLPPolicy, QNetwork
from methods.networks.targets import TargetActor, TargetCritic

from utils.buffer import ReplayBuffer, StratLastRewards


class DDPG(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        hidden_dim=256,
    ):
        super(DDPG, self).__init__()
        self.gamma = args.gamma
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = np.array(observation_space.shape).prod()
        self.num_actions = np.array(action_space.shape).prod()
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.actor, self.critic = self.get_networks(
            hidden_dim,
        )
        self.actor_target = TargetActor(self.actor)
        self.critic_target = TargetCritic(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)

        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def get_networks(
        self,
        hidden_dim=256,
    ):
        actor = MLPPolicy(
            self.num_inputs,
            self.num_actions,
            hidden_dim=hidden_dim,
        )
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
        self.actor_target.target_model.to(device)
        return super(DDPG, self).to(device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        action = self.actor.get_action(state)
        return action

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            qf_next_target = self.critic_target(next_state_batch, next_action)
            qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * qf_next_target

        qf = self.critic(state_batch, action_batch)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = F.mse_loss(qf, next_q_value)

        # Minimize the loss between two Q-functions
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        return qf_loss

    def update_actor(self, state_batch):
        pi = self.actor(state_batch)
        qf_pi = self.critic(state_batch, pi)
        policy_loss = -qf_pi.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        return policy_loss

    def update(self, batch_size, update_actor=False):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)

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


class DDPGStrat(DDPG):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        hidden_dim=256,
    ):
        self.is_dylam = args.dylam
        self.num_rewards = args.num_rewards
        super().__init__(args, observation_space, action_space, hidden_dim)
        if args.dylam:
            self.lambdas = (
                torch.ones(args.num_rewards).to(self.device) / args.num_rewards
            )
        else:
            self.lambdas = torch.ones(args.num_rewards).to(self.device)
        self.r_max = torch.Tensor(args.r_max).to(self.device)
        self.r_min = torch.Tensor(args.r_min).to(self.device)
        self.rew_tau = args.dylam_tau
        self.episode_rewards = np.zeros((args.num_envs, args.num_rewards))
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def get_networks(
        self,
        hidden_dim=256,
    ):
        actor = MLPPolicy(
            self.num_inputs,
            self.num_actions,
            hidden_dim=hidden_dim,
        )
        critic = QNetwork(
            self.num_inputs,
            self.num_actions,
            hidden_dim=hidden_dim,
            num_outputs=self.num_rewards,
        )
        return actor, critic

    def update_actor(self, state_batch):
        pi = self.actor(state_batch)
        qf_pi = self.critic(state_batch, pi)
        qf_pi = torch.einsum("ij,j->i", qf_pi, self.lambdas).view(-1, 1)
        policy_loss = -qf_pi.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        return policy_loss

    def update_lambdas(self):
        if self.last_episode_rewards.can_do():
            rew_mean_t = torch.Tensor(self.last_episode_rewards.mean()).to(self.device)
            if self.last_reward_mean is not None:
                rew_mean_t = (
                    rew_mean_t + (self.last_reward_mean - rew_mean_t) * self.rew_tau
                )
            dQ = torch.clamp(
                (self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1
            )
            expdQ = torch.exp(dQ) - 1
            self.lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
            self.last_reward_mean = rew_mean_t

    def add_episode_rewards(self, rewards, terminations, truncations):
        if self.num_rewards == 1:
            rewards = rewards.reshape(-1, 1)
        self.episode_rewards += rewards
        for i, (term, trunc) in enumerate(zip(terminations, truncations)):
            if term or trunc:
                self.last_episode_rewards.add(self.episode_rewards[i])
                self.episode_rewards[i] = np.zeros(self.num_rewards)
