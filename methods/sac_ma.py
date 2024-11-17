import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from methods.networks.architectures import GaussianPolicy, DoubleQNetwork
from methods.networks.targets import TargetCritic

from utils.buffer import ReplayBuffer, StratLastRewards


class SACMA(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
    ):
        super(SACMA, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.n_hidden = args.n_hidden
        self.num_rewards = args.num_rewards
        self.multiple_policies = args.multiple_policies
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = np.array(observation_space["agent_1"].shape).prod()
        self.num_actions = np.array(action_space["agent_1"].shape).prod()
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.actor, self.critic = self.get_networks()
        self.critic_target = TargetCritic(self.critic)
        self.set_optimizers(args)
        self.set_sac_tools(args)
        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def set_optimizers(self, args):
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)
        if self.multiple_policies:
            self.actor_optim = {
                f"agent_{i}": Adam(self.actor.parameters(), lr=args.policy_lr)
                for i in range(1, len(self.observation_space) + 1)
            }
        else:
            self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)

    def set_sac_tools(self, args):
        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optim = None
        if args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_space["agent_1"].shape).to(self.device)
            ).item()
            if args.multiple_policies:
                self.log_alpha = {
                    f"agent_{i}": torch.zeros(1, requires_grad=True, device=self.device)
                    for i in range(1, len(self.observation_space) + 1)
                }
                self.alpha = {
                    f"agent_{i}": self.log_alpha[f"agent_{i}"].exp().item()
                    for i in range(1, len(self.observation_space) + 1)
                }
                self.alpha_optim = {
                    f"agent_{i}": Adam(
                        [self.log_alpha[f"agent_{i}"]], lr=args.policy_lr
                    )
                    for i in range(1, len(self.observation_space) + 1)
                }
            else:
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp().item()
                self.alpha_optim = Adam([self.log_alpha], lr=args.policy_lr)
        else:
            self.alpha = args.alpha

    def get_networks(self):
        if self.multiple_policies:
            actor = {
                f"agent_{i}": GaussianPolicy(
                    self.num_inputs,
                    self.num_actions,
                    log_sig_min=self.log_sig_min,
                    log_sig_max=self.log_sig_max,
                    n_hidden=1,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                for i in range(1, len(self.observation_space) + 1)
            }
        else:
            actor = GaussianPolicy(
                self.num_inputs,
                self.num_actions,
                log_sig_min=self.log_sig_min,
                log_sig_max=self.log_sig_max,
                n_hidden=1,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
        critic = DoubleQNetwork(
            self.num_inputs,
            self.num_actions,
            num_outputs=self.num_rewards,
            n_hidden=self.n_hidden,
        )
        return actor, critic

    def get_replay_buffer(self, buffer_size):
        return ReplayBuffer(buffer_size, self.device)

    def to(self, device):
        if self.multiple_policies:
            for actor in self.actor.values():
                actor.to(device)
        else:
            self.actor.to(device)
        self.critic.to(device)
        self.critic_target.target_model.to(device)
        return super(SACMA, self).to(device)

    def get_action(self, state):
        states = [
            torch.Tensor(state[f"agent_{i}"]).to(self.device)
            for i in range(1, len(self.observation_space) + 1)
        ]
        if self.multiple_policies:
            actions = {
                f"agent_{i+1}": self.actor[f"agent_{i}"].get_action(states[i])
                for i in range(len(states))
            }
        else:
            actions = {
                f"agent_{i+1}": self.actor.get_action(states[i])
                for i in range(len(states))
            }
        return actions

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        with torch.no_grad():
            if self.multiple_policies:
                next_sample = self.actor[f"agent_{np.random.choice(2)+1}"].sample(
                    next_state_batch
                )
            else:
                next_sample = self.actor.sample(next_state_batch)
            next_state_action, next_state_log_pi, next_state_action_probs = next_sample
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
            min_qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target
        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)

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
            if self.multiple_policies:
                losses = {}
                for i in range(1, len(self.observation_space) + 1):
                    with torch.no_grad():
                        _, log_pi, _ = self.actor[f"agent_{i}"].sample(state_batch)
                    alpha_loss = (
                        -self.log_alpha[f"agent_{i}"]
                        * (log_pi + self.target_entropy).detach()
                    )
                    alpha_loss = alpha_loss.mean()

                    self.alpha_optim[f"agent_{i}"].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim[f"agent_{i}"].step()
                    self.alpha[f"agent_{i}"] = self.log_alpha[f"agent_{i}"].exp().item()
                    losses[f"agent_{i}"] = alpha_loss.item()
                alpha_loss = np.mean(list(losses.values()))
            else:
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
        if self.multiple_policies:
            losses = {}
            for i in range(1, len(self.observation_space) + 1):
                pi, log_pi, action_probs = self.actor[f"agent_{i}"].sample(state_batch)
                qf1_pi, qf2_pi = self.critic(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                policy_loss = self.alpha * log_pi
                policy_loss = policy_loss - min_qf_pi
                policy_loss = policy_loss.mean()

                self.actor_optim[f"agent_{i}"].zero_grad()
                policy_loss.backward()
                self.actor_optim[f"agent_{i}"].step()
                losses[f"agent_{i}"] = policy_loss.item()
            policy_loss = np.mean(list(losses.values()))
        else:
            pi, log_pi, action_probs = self.actor.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = self.alpha * log_pi
            policy_loss = policy_loss - min_qf_pi
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
        ) = self.replay_buffer.sample(batch_size)

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
        if self.multiple_policies:
            for i in range(1, len(self.observation_space) + 1):
                torch.save(
                    self.actor[f"agent_{i}"].state_dict(),
                    path + f"actor_agent_{i}.pt",
                )
        else:
            torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path):
        if self.multiple_policies:
            for i in range(1, len(self.observation_space) + 1):
                self.actor[f"agent_{i}"].load_state_dict(
                    torch.load(path + f"actor_agent_{i}.pt", map_location=self.device)
                )
        else:
            self.actor.load_state_dict(
                torch.load(path + "actor.pt", map_location=self.device)
            )
        self.critic.load_state_dict(
            torch.load(path + "critic.pt", map_location=self.device)
        )


class SACStratMA(SACMA):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
    ):
        super().__init__(
            args, observation_space, action_space, log_sig_min, log_sig_max
        )
        self.lambdas = torch.ones(args.num_rewards).to(self.device) / args.num_rewards
        self.r_max = torch.Tensor(args.r_max).to(self.device)
        self.r_min = torch.Tensor(args.r_min).to(self.device)
        self.rew_tau = args.dylam_tau
        self.episode_rewards = np.zeros((args.num_envs, args.num_rewards))
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def update_actor(self, state_batch):
        if self.multiple_policies:
            losses = {}
            for i in range(1, len(self.observation_space) + 1):
                pi, log_pi, action_probs = self.actor[f"agent_{i}"].sample(state_batch)
                qf1_pi, qf2_pi = self.critic(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                min_qf_pi = torch.einsum("ij,j->i", min_qf_pi, self.lambdas).view(-1, 1)

                # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                policy_loss = self.alpha * log_pi
                policy_loss = policy_loss - min_qf_pi
                policy_loss = policy_loss.mean()

                self.actor_optim[f"agent_{i}"].zero_grad()
                policy_loss.backward()
                self.actor_optim[f"agent_{i}"].step()
                losses[f"agent_{i}"] = policy_loss.item()
            policy_loss = np.mean(list(losses.values()))
        else:
            pi, log_pi, _ = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            min_qf_pi = torch.einsum("ij,j->i", min_qf_pi, self.lambdas).view(-1, 1)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = self.alpha * log_pi
            policy_loss = policy_loss - min_qf_pi
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
        ) = self.replay_buffer.sample(batch_size)

        reward_batch = reward_batch * self.reward_scaling
        qf1_loss, qf2_loss = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch)

        return policy_loss, qf1_loss, qf2_loss, alpha_loss

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
