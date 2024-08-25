import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.optim import Adam
from methods.networks.architectures import GaussianPolicy, DoubleQNetwork
from methods.networks.targets import TargetCritic

from utils.gpi_ls import unique_tol
from utils.buffer import ReplayBuffer


class SACGPILS(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
    ):
        super(SACGPILS, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.n_hidden = args.n_hidden
        self.num_rewards = args.num_rewards
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = np.array(observation_space.shape).prod()
        self.num_actions = np.array(action_space.shape).prod()
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.actor, self.critic = self.get_networks()
        self.critic_target = TargetCritic(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.q_lr)

        # Automatic entropy tuning
        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optim = None
        if args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.weight_support = []
        self.stacked_weight_support = []
        self.max_iterations = args.total_timesteps // args.steps_per_iteration

        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def get_networks(self):
        actor = GaussianPolicy(
            self.num_inputs + self.num_rewards,
            self.num_actions,
            log_sig_min=self.log_sig_min,
            log_sig_max=self.log_sig_max,
            n_hidden=1,
            epsilon=self.epsilon,
            action_space=self.action_space,
        )
        critic = DoubleQNetwork(
            self.num_inputs + self.num_rewards,
            self.num_actions,
            num_outputs=self.num_rewards,
            n_hidden=self.n_hidden,
        )
        return actor, critic

    def get_replay_buffer(self, buffer_size):
        return ReplayBuffer(buffer_size, self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.target_model.to(device)
        return super().to(device)

    def get_action(self, state, lambdas=None):
        state = torch.Tensor(state).to(self.device)
        lambdas = self.current_lambdas if lambdas is None else lambdas
        lambdas = lambdas.repeat(state.size(0), 1)
        action = self.actor.get_action(state, lambdas)
        return action

    def set_weight_support(self, weight_list):
        """Set the weight support set."""
        weights_no_repeat = unique_tol(weight_list)
        self.weight_support = [
            torch.tensor(w).float().to(self.device) for w in weights_no_repeat
        ]
        if len(self.weight_support) > 0:
            self.stacked_weight_support = torch.stack(self.weight_support)

    def set_current_lambdas(self, current_lambda):
        self.current_lambdas = torch.tensor(current_lambda).float().to(self.device)

    def update_critic(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        lambdas_batch,
    ):
        with torch.no_grad():
            next_state_action, next_state_log_pi, next_state_action_probs = (
                self.actor.sample(next_state_batch, lambdas_batch)
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action, lambdas_batch
            )

            qf1_next_target *= lambdas_batch[0]
            qf2_next_target *= lambdas_batch[0]
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
            min_qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch, lambdas_batch)

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

    def update_alpha(self, state_batch, lambdas_batch):
        alpha_loss = None
        if self.alpha_optim is not None:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state_batch, lambdas_batch)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
            alpha_loss = alpha_loss.mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        return alpha_loss

    def update_actor(self, state_batch, lambdas_batch):
        pi, log_pi, action_probs = self.actor.sample(state_batch, lambdas_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, lambdas_batch)
        qf1_pi = torch.einsum("ij,j->i", qf1_pi, lambdas_batch[0]).view(-1, 1)
        qf2_pi = torch.einsum("ij,j->i", qf2_pi, lambdas_batch[0]).view(-1, 1)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch, lambdas_batch)

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

        lambdas_batch = self.current_lambdas.repeat(state_batch.size(0), 1)
        if len(self.weight_support) > 1:
            lambdas_batch = lambdas_batch[: state_batch.size(0) // 2]
            from_support = random.choices(
                self.weight_support, k=state_batch.size(0) // 2
            )
            from_support = torch.stack(from_support)
            lambdas_batch = torch.cat([lambdas_batch, from_support])
        lambdas_batch = lambdas_batch.to(self.device)

        qf1_loss, qf2_loss = self.update_critic(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            lambdas_batch,
        )
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch, lambdas_batch)

        return policy_loss, qf1_loss, qf2_loss, alpha_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path):
        self.actor.load_state_dict(
            torch.load(path + "actor.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(path + "critic.pt", map_location=self.device)
        )
        self.actor.eval()
        self.critic.eval()
