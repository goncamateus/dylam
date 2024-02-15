import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from methods.networks.image.discrete import ImageCategoricalPolicy, ImageQNetwork
from methods.networks.raw.continuous import GaussianPolicy, QNetwork
from methods.networks.raw.discrete import CategoricalPolicy, DiscreteQNetwork
from methods.networks.targets import TargetCritic

from utils.buffer import ReplayBuffer, StratLastRewards


class SAC(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
    ):
        super(SAC, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.with_image = args.with_image
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.continuous_actions = not isinstance(action_space, gym.spaces.Discrete)
        if self.with_image:
            self.num_inputs = observation_space.shape
        else:
            self.num_inputs = np.array(observation_space.shape).prod()
        if self.continuous_actions:
            self.num_actions = np.array(action_space.shape).prod()
        else:
            self.num_actions = action_space.n
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.actor, self.critic = self.get_networks(
            action_space,
            args.epsilon,
            log_sig_min,
            log_sig_max,
            hidden_dim,
            self.with_image,
        )
        self.critic_target = TargetCritic(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)

        # Automatic entropy tuning
        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optim = None
        if args.autotune:
            if self.continuous_actions:
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.action_space.shape).to(self.device)
                ).item()
            else:
                self.target_entropy = -args.target_entropy_scale * torch.log(
                    1 / torch.tensor(self.num_actions)
                ).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def get_networks(
        self,
        action_space,
        epsilon,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        with_image=False,
    ):
        if with_image:
            actor = ImageCategoricalPolicy(
                self.observation_space.shape,
                self.num_actions,
                log_sig_min=log_sig_min,
                log_sig_max=log_sig_max,
                hidden_dim=hidden_dim,
                epsilon=epsilon,
                action_space=action_space,
            )
            critic = ImageQNetwork(
                self.observation_space.shape,
                self.num_actions,
                hidden_dim=hidden_dim,
            )
        else:
            if self.continuous_actions:
                actor = GaussianPolicy(
                    self.num_inputs,
                    self.num_actions,
                    log_sig_min=log_sig_min,
                    log_sig_max=log_sig_max,
                    hidden_dim=hidden_dim,
                    epsilon=epsilon,
                    action_space=action_space,
                )
                critic = QNetwork(
                    self.num_inputs,
                    self.num_actions,
                    hidden_dim=hidden_dim,
                )
            else:
                actor = CategoricalPolicy(
                    self.num_inputs,
                    self.num_actions,
                    hidden_dim=hidden_dim,
                    action_space=action_space,
                )
                critic = DiscreteQNetwork(
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
        return super(SAC, self).to(device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor.get_action(state)

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        with torch.no_grad():
            next_state_action, next_state_log_pi, next_state_action_probs = (
                self.actor.sample(next_state_batch)
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            if not self.continuous_actions:
                next_state_log_pi = next_state_log_pi.unsqueeze(2)
                next_state_action_probs = next_state_action_probs.unsqueeze(2)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
            min_qf_next_target *= 1 if self.continuous_actions else next_state_action_probs
            
            min_qf_next_target[done_batch] = 0.0
            if not self.continuous_actions:
                min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = reward_batch + self.gamma * min_qf_next_target
        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        if self.continuous_actions:
            qf1, qf2 = self.critic(state_batch, action_batch)
        else:
            qf1, qf2 = self.critic(state_batch)
            indices = action_batch.squeeze(1)
            qf1 = qf1[torch.arange(qf1.size(0)), indices, :].squeeze()
            qf2 = qf2[torch.arange(qf2.size(0)), indices, :].squeeze()
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)

        # Minimize the loss between two Q-functions
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        return qf1_loss, qf2_loss

    def update_alpha(self, state_batch):
        alpha_loss = None
        if self.alpha_optim is not None:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state_batch)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
            alpha_loss = alpha_loss.mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        return alpha_loss

    def update_actor(self, state_batch):
        pi, log_pi, action_probs = self.actor.sample(state_batch)
        if self.continuous_actions:
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
        else:
            qf1_pi, qf2_pi = self.critic(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        if self.continuous_actions:
            policy_loss = policy_loss - min_qf_pi
        else:
            policy_loss = (action_probs * policy_loss) - min_qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch)

        return policy_loss, alpha_loss

    def update(self, batch_size, update_actor=False):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size, continuous=self.continuous_actions)

        reward_batch = reward_batch * self.reward_scaling
        qf1_loss, qf2_loss = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch)

        return policy_loss, qf1_loss, qf2_loss, alpha_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.critic.load_state_dict(torch.load(path + "critic.pt"))
        self.actor.eval()
        self.critic.eval()


class SACStrat(SAC):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
    ):
        self.is_dylam = args.dylam
        self.num_rewards = args.num_rewards
        super().__init__(
            args, observation_space, action_space, log_sig_min, log_sig_max, hidden_dim
        )
        if args.dylam:
            self.lambdas = (
                torch.ones(args.num_rewards).to(self.device) / args.num_rewards
            )
        else:
            self.lambdas = torch.Tensor(args.lambdas).to(self.device)
        self.r_max = torch.Tensor(args.r_max).to(self.device)
        self.r_min = torch.Tensor(args.r_min).to(self.device)
        self.rew_tau = args.dylam_tau
        self.episode_rewards = np.zeros((args.num_envs, args.num_rewards))
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def get_networks(
        self,
        action_space,
        epsilon,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        with_image=False,
    ):
        if with_image:
            actor = ImageCategoricalPolicy(
                self.observation_space.shape,
                self.num_actions,
                log_sig_min=log_sig_min,
                log_sig_max=log_sig_max,
                hidden_dim=hidden_dim,
                epsilon=epsilon,
                action_space=action_space,
            )
            critic = ImageQNetwork(
                self.observation_space.shape,
                self.num_actions,
                hidden_dim=hidden_dim,
                num_outputs=self.num_rewards,
            )
        else:
            if self.continuous_actions:
                actor = GaussianPolicy(
                    self.num_inputs,
                    self.num_actions,
                    log_sig_min=log_sig_min,
                    log_sig_max=log_sig_max,
                    hidden_dim=hidden_dim,
                    epsilon=epsilon,
                    action_space=action_space,
                )
                critic = QNetwork(
                    self.num_inputs,
                    self.num_actions,
                    hidden_dim=hidden_dim,
                    num_outputs=self.num_rewards,
                )
            else:
                actor = CategoricalPolicy(
                    self.num_inputs,
                    self.num_actions,
                    hidden_dim=hidden_dim,
                    action_space=action_space,
                )
                critic = DiscreteQNetwork(
                    self.num_inputs,
                    self.num_actions,
                    hidden_dim=hidden_dim,
                    num_outputs=self.num_rewards,
                )
        return actor, critic

    def update_actor(self, state_batch):
        pi, log_pi, action_probs = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = (
            self.critic(state_batch, pi)
            if self.continuous_actions
            else self.critic(state_batch)
        )
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        if self.continuous_actions:
            min_qf_pi = torch.einsum("ij,j->i", min_qf_pi, self.lambdas).view(-1, 1)
        else:
            min_qf_pi = torch.einsum("ijk,k->ij", min_qf_pi, self.lambdas)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        if not self.continuous_actions:
            policy_loss *= action_probs
        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch)

        return policy_loss, alpha_loss

    def update_lambdas(self):
        rew_mean_t = torch.Tensor(self.last_episode_rewards.mean()).to(self.device)
        if self.last_reward_mean is not None:
            rew_mean_t = (
                rew_mean_t + (self.last_reward_mean - rew_mean_t) * self.rew_tau
            )
        dQ = torch.clamp((self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1)
        expdQ = torch.exp(dQ) - 1
        self.lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
        self.last_reward_mean = rew_mean_t

    def add_episode_rewards(self, rewards, terminations, truncations):
        self.episode_rewards += rewards
        for i, (term, trunc) in enumerate(zip(terminations, truncations)):
            if term or trunc:
                self.last_episode_rewards.add(self.episode_rewards[i])
                self.episode_rewards[i] = np.zeros(self.num_rewards)
