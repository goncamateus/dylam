"""
Proximal Policy Optimization (PPO) implementation for reinforcement learning.

This module defines the PPO class, which implements the PPO algorithm for training
policy and value networks in environments with discrete or continuous action spaces.
It supports multi-reward settings, advantage normalization, entropy regularization,
and KL-based early stopping. The class is compatible with both standard and MiniGrid
network architectures and uses a replay buffer for efficient minibatch updates.

Classes:
    PPO: Main class implementing the PPO algorithm.
"""

import os
from collections import namedtuple
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from dylam.methods.networks.architectures import MLPPolicy, QNetwork
from dylam.methods.networks.mini_grid.architectures import Policy as MiniGridPolicy
from dylam.methods.networks.mini_grid.architectures import QNetwork as MiniGridQNetwork
from dylam.utils.buffer import StratLastRewards
from dylam.utils.experiment import l1_norm, minmax_norm, softmax_norm

buffer = namedtuple(
    "buffer", ["states", "actions", "logprobs", "rewards", "dones", "values"]
)


class PPO(nn.Module):
    def __init__(
        self,
        args: Any,
        observation_space: Any,
        action_space: Any,
    ) -> None:
        """
        Initialize the PPO agent.

        Args:
            args: Namespace or object containing hyperparameters and settings.
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
        """
        super(PPO, self).__init__()
        self.batch_size = int(args.num_envs * args.num_steps)
        self.minibatch_size = args.batch_size // args.num_minibatches
        self.update_epochs = args.update_epochs
        self.num_mini_batches = args.num_minibatches
        self.epsilon = args.epsilon
        self.gamma: float = args.gamma
        self.n_hidden = args.n_hidden
        self.num_rewards = args.num_rewards
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = int(np.array(observation_space.shape).prod())
        self.num_actions = action_space.n
        self.reward_scaling = args.reward_scaling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        self.entropy_coef: float = args.entropy_coef
        self.clip_qf_loss: bool = args.clip_qf_loss
        self.clip_coef: bool = args.clip_coef
        self.max_grad_norm: float = args.max_grad_norm
        self.norm_adv: bool = args.norm_adv
        self.qf_coef: float = args.qf_coef
        self.target_kl: float = args.target_kl
        self.num_steps: int = args.num_steps
        self.num_envs = args.num_envs
        self.gae_lambda: float = args.gae_lambda

        self.actor, self.critic = self.get_networks()
        self.actor_optim = Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.learning_rate)

        self.replay_buffer: buffer = self.set_buffer(args.num_envs)
        self.to(self.device)

    def set_buffer(self, num_envs: int) -> buffer:
        states = torch.zeros(
            (self.num_steps, num_envs) + self.observation_space.shape
        ).to(self.device)
        actions = torch.zeros((self.num_steps, num_envs) + self.action_space.shape).to(
            self.device
        )
        logprobs = torch.zeros((self.num_steps, num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, num_envs)).to(self.device)
        return buffer(states, actions, logprobs, rewards, dones, values)

    def add_to_buffer(
        self,
        step: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.replay_buffer.states[step] = states
        self.replay_buffer.actions[step] = actions
        self.replay_buffer.logprobs[step] = logprobs
        self.replay_buffer.rewards[step] = rewards.view(-1)
        self.replay_buffer.dones[step] = dones
        self.replay_buffer.values[step] = values

    def get_networks(self) -> Tuple[nn.Module, nn.Module]:
        """
        Create actor and critic networks based on the observation and action spaces.

        Returns:
            actor: Policy network.
            critic: Value network.
        """

        def q_net():
            q_net = QNetwork(
                self.num_inputs,
                num_actions=0,
                num_outputs=self.num_rewards,
                n_hidden=self.n_hidden,
            )
            if len(self.observation_space.shape) == 3:
                q_net = MiniGridQNetwork(
                    self.observation_space,
                    num_actions=0,
                    num_out_features=128,
                    num_outputs=self.num_rewards,
                    num_hidden=self.n_hidden,
                )
            return q_net

        def policy():
            policy = MLPPolicy(
                self.num_inputs,
                self.num_actions,
                n_hidden=self.n_hidden,
            )
            if len(self.observation_space.shape) == 3:
                policy = MiniGridPolicy(
                    self.observation_space,
                    num_actions=self.num_actions,
                    num_outputs=128,
                    n_hidden=self.n_hidden,
                )
            return policy

        critic = q_net()
        actor = policy()
        return actor, critic

    def to(self, device: torch.device) -> "PPO":
        """
        Move the model's parameters to the specified device.

        Args:
            device: Target device (e.g., 'cpu' or 'cuda').

        Returns:
            self
        """
        self.actor.to(device)
        self.critic.to(device)
        return super(PPO, self).to(device)

    def get_policy_returns(
        self, state: Any, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the action, log probability, and entropy for a given state.

        Args:
            state: Input state(s).
            action: Optional action(s) to evaluate log prob and entropy.

        Returns:
            action: Sampled or provided action(s).
            log_prob: Log probability of the action(s).
            entropy: Entropy of the action distribution.
        """
        state = torch.Tensor(state).to(self.device)
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action.cpu().long().numpy(),
            probs.log_prob(action.squeeze()),
            probs.entropy(),
        )

    def get_value(self, state: Any) -> torch.Tensor:
        """
        Compute the value estimate for a given state.

        Args:
            state: Input state(s).

        Returns:
            value: Estimated value(s) of the state(s).
        """
        state = torch.Tensor(state).to(self.device)
        value = self.critic(
            state,
            torch.Tensor([]).to(self.device),
            torch.Tensor([]).to(self.device),
        )
        return value

    def compute_returns_and_advantages(
        self,
        next_obs: np.ndarray,
        next_done: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and Generalized Advantage Estimation (GAE).

        Args:
            next_obs: Next observation(s).
            next_done: Done flags for next observation(s).

        Returns:
            returns: Computed returns.
            advantages: Computed advantages.
        """
        rewards = self.replay_buffer.rewards * self.reward_scaling
        with torch.no_grad():
            next_done = torch.Tensor(next_done).to(self.device)
            next_value = self.get_value(next_obs).squeeze()
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                else:
                    next_non_terminal = 1.0 - self.replay_buffer.dones[t + 1].float()
                    next_value = self.replay_buffer.values[t + 1]
                delta = (
                    rewards[t]
                    + self.gamma * next_value * next_non_terminal
                    - self.replay_buffer.values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )
            returns = advantages + self.replay_buffer.values
        return returns, advantages

    def update_critic(
        self,
        state_batch: torch.Tensor,
        return_batch: torch.Tensor,
        value_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the critic (value function) loss for a minibatch.

        Args:
            state_batch: Batch of states.
            return_batch: Batch of target returns.
            value_batch: Batch of predicted values.

        Returns:
            qf_loss: Critic loss value.
        """
        new_value = self.get_value(state_batch).view(-1)
        if self.clip_qf_loss:
            qf_loss_unclipped = (new_value - return_batch) ** 2
            qf_clipped = value_batch + torch.clamp(
                new_value - value_batch,
                -self.clip_coef,
                self.clip_coef,
            )
            qf_loss_clipped = (qf_clipped - return_batch) ** 2
            qf_loss_max = torch.max(qf_loss_unclipped, qf_loss_clipped)
            qf_loss = 0.5 * qf_loss_max.mean()
        else:
            qf_loss = 0.5 * ((new_value - return_batch) ** 2).mean()

        return qf_loss

    def update_actor(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        logprob_batch: torch.Tensor,
        advantage_batch: torch.Tensor,
        clipfracs: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the actor (policy) loss for a minibatch.

        Args:
            state_batch: Batch of states.
            action_batch: Batch of actions.
            logprob_batch: Batch of log probabilities.
            advantage_batch: Batch of advantages.
            clipfracs: List to accumulate clipping fractions.

        Returns:
            policy_loss: Policy loss value.
            entropy_loss: Entropy loss value.
            approx_kl: Approximate KL divergence.
        """
        _, newlogprob, entropy = self.get_policy_returns(state_batch, action_batch)
        logratio = newlogprob - logprob_batch
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

        if self.norm_adv:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

        # Policy loss
        policy_loss1 = -advantage_batch.squeeze() * ratio
        policy_loss2 = -advantage_batch.squeeze() * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        entropy_loss = entropy.mean()
        return policy_loss, entropy_loss, approx_kl

    def update_minibatch(
        self,
        minibatch: Tuple[torch.Tensor, ...],
        clipfracs: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update actor and critic using a single minibatch.

        Args:
            minibatch: Tuple of minibatch data.
            clipfracs: List to accumulate clipping fractions.

        Returns:
            qf_loss: Critic loss value.
            policy_loss: Policy loss value.
            entropy_loss: Entropy loss value.
            loss: Total loss value.
            approx_kl: Approximate KL divergence.
        """
        (
            state_batch,
            log_prob_batch,
            action_batch,
            advantage_batch,
            return_batch,
            value_batch,
        ) = minibatch

        qf_loss = self.update_critic(state_batch, return_batch, value_batch)
        policy_loss, entropy_loss, approx_kl = self.update_actor(
            state_batch,
            action_batch,
            log_prob_batch,
            advantage_batch,
            clipfracs,
        )
        loss = policy_loss - self.entropy_coef * entropy_loss + qf_loss * self.qf_coef

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        self.critic_optim.step()

        return qf_loss, policy_loss, entropy_loss, loss, approx_kl

    def get_batch(self, advantages, returns) -> Tuple[torch.Tensor, ...]:
        b_obs = self.replay_buffer.states.reshape((-1,) + self.observation_space.shape)
        b_logprobs = self.replay_buffer.logprobs.reshape(-1)
        b_actions = self.replay_buffer.actions.reshape((-1,) + self.action_space.shape)
        b_values = self.replay_buffer.values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def update(
        self,
        next_obs: np.ndarray,
        next_done: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a full PPO update using all minibatches and epochs.

        Args:
            next_obs: Last observation(s) after rollout.
            next_done: Done flags for last observation(s).

        Returns:
            loss_mean: Mean total loss over epochs.
            policy_loss_mean: Mean policy loss.
            qf_loss_mean: Mean critic loss.
            entropy_loss_mean: Mean entropy loss.
        """
        clipfracs = []
        policy_loss_mean = 0
        qf_loss_mean = 0
        entropy_loss_mean = 0
        loss_mean = 0
        returns, advantages = self.compute_returns_and_advantages(next_obs, next_done)

        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
            self.get_batch(returns=returns, advantages=advantages)
        )
        b_inds = np.arange(self.batch_size)
        for epoch in range(self.update_epochs):
            loss_epoch = 0
            policy_loss_epoch = 0
            qf_loss_epoch = 0
            entropy_loss_epoch = 0
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                minibatch = (
                    b_obs[mb_inds],
                    b_logprobs[mb_inds],
                    b_actions[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                )
                (
                    qf_loss,
                    policy_loss,
                    entropy_loss,
                    loss,
                    approx_kl,
                ) = self.update_minibatch(minibatch, clipfracs)

                loss_epoch += loss
                policy_loss_epoch += policy_loss
                qf_loss_epoch += qf_loss
                entropy_loss_epoch += entropy_loss

            policy_loss_mean += policy_loss_epoch / self.num_mini_batches
            qf_loss_mean += qf_loss_epoch / self.num_mini_batches
            entropy_loss_mean += entropy_loss_epoch / self.num_mini_batches
            loss_mean += loss_epoch / self.num_mini_batches

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        policy_loss_mean /= self.update_epochs
        qf_loss_mean /= self.update_epochs
        entropy_loss_mean /= self.update_epochs
        loss_mean /= self.update_epochs
        return loss_mean, policy_loss_mean, qf_loss_mean, entropy_loss_mean

    def save(self, path: str) -> None:
        """
        Save the actor and critic model parameters to the specified path.

        Args:
            path: Directory path to save model files.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path: str) -> None:
        """
        Load the actor and critic model parameters from the specified path.

        Args:
            path: Directory path to load model files from.
        """
        self.actor.load_state_dict(
            torch.load(
                path + "actor.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                path + "critic.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.actor.eval()
        self.critic.eval()


class PPOStrat(PPO):
    def __init__(self, args, observation_space, action_space):
        super().__init__(args, observation_space, action_space)
        self.lambdas = torch.ones(args.num_rewards).to(self.device) / args.num_rewards
        self.r_max = torch.Tensor(args.r_max).to(self.device)
        self.r_min = torch.Tensor(args.r_min).to(self.device)
        self.normalizer = self.set_normalizer(args.normalizer)
        self.rew_tau = args.dylam_tau
        self.episode_rewards = np.zeros((args.num_envs, args.num_rewards))
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def set_normalizer(self, normalizer):
        norm_func = None
        if normalizer == "softmax":
            norm_func = softmax_norm
        elif normalizer == "minmax":
            norm_func = minmax_norm
        elif normalizer == "l1":
            norm_func = l1_norm
        else:
            raise ValueError(f"Normalizer {normalizer} not found.")
        return norm_func

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
            self.lambdas = self.normalizer(dQ)
            self.last_reward_mean = rew_mean_t

    def add_episode_rewards(self, rewards, terminations, truncations):
        if self.num_rewards == 1:
            rewards = rewards.reshape(-1, 1)
        self.episode_rewards += rewards
        for i, (term, trunc) in enumerate(zip(terminations, truncations)):
            if term or trunc:
                self.last_episode_rewards.add(self.episode_rewards[i])
                self.episode_rewards[i] = np.zeros(self.num_rewards)

    def set_buffer(self, num_envs):
        states = torch.zeros(
            (self.num_steps, num_envs) + self.observation_space.shape
        ).to(self.device)
        actions = torch.zeros((self.num_steps, num_envs) + self.action_space.shape).to(
            self.device
        )
        logprobs = torch.zeros((self.num_steps, num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, num_envs, self.num_rewards)).to(
            self.device
        )
        dones = torch.zeros((self.num_steps, num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, num_envs, self.num_rewards)).to(
            self.device
        )
        return buffer(states, actions, logprobs, rewards, dones, values)

    def add_to_buffer(
        self,
        step: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.replay_buffer.states[step] = states
        self.replay_buffer.actions[step] = actions
        self.replay_buffer.logprobs[step] = logprobs
        self.replay_buffer.rewards[step] = rewards
        self.replay_buffer.dones[step] = dones
        self.replay_buffer.values[step] = values

    def compute_returns_and_advantages(
        self,
        next_obs: np.ndarray,
        next_done: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and Generalized Advantage Estimation (GAE).

        Args:
            next_obs: Next observation(s).
            next_done: Done flags for next observation(s).

        Returns:
            returns: Computed returns.
            advantages: Computed advantages.
        """
        rewards = self.replay_buffer.rewards * self.reward_scaling
        with torch.no_grad():
            next_done = torch.Tensor(next_done).to(self.device)
            next_value = self.get_value(next_obs).squeeze()
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                else:
                    next_non_terminal = 1.0 - self.replay_buffer.dones[t + 1].float()
                    next_value = self.replay_buffer.values[t + 1]
                next_non_terminal = next_non_terminal.unsqueeze(-1)
                delta = (
                    rewards[t]
                    + self.gamma * next_value * next_non_terminal
                    - self.replay_buffer.values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )
            returns = advantages + self.replay_buffer.values
        return returns, advantages

    def update_actor(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        logprob_batch: torch.Tensor,
        advantage_batch: torch.Tensor,
        clipfracs: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the actor (policy) loss for a minibatch.

        Args:
            state_batch: Batch of states.
            action_batch: Batch of actions.
            logprob_batch: Batch of log probabilities.
            advantage_batch: Batch of advantages.
            clipfracs: List to accumulate clipping fractions.

        Returns:
            policy_loss: Policy loss value.
            entropy_loss: Entropy loss value.
            approx_kl: Approximate KL divergence.
        """
        _, newlogprob, entropy = self.get_policy_returns(state_batch, action_batch)
        logratio = newlogprob - logprob_batch
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

        if self.norm_adv:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

        advantage_batch = (advantage_batch * self.lambdas).sum(dim=1)
        # Policy loss
        policy_loss1 = -advantage_batch.squeeze() * ratio
        policy_loss2 = -advantage_batch.squeeze() * torch.clamp(
            ratio, 1 - self.clip_coef, 1 + self.clip_coef
        )
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        entropy_loss = entropy.mean()
        return policy_loss, entropy_loss, approx_kl

    def update_critic(
        self,
        state_batch: torch.Tensor,
        return_batch: torch.Tensor,
        value_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the critic (value function) loss for a minibatch.

        Args:
            state_batch: Batch of states.
            return_batch: Batch of target returns.
            value_batch: Batch of predicted values.

        Returns:
            qf_loss: Critic loss value.
        """
        new_value = self.get_value(state_batch)
        if self.clip_qf_loss:
            qf_loss_unclipped = (new_value - return_batch) ** 2
            qf_clipped = value_batch + torch.clamp(
                new_value - value_batch,
                -self.clip_coef,
                self.clip_coef,
            )
            qf_loss_clipped = (qf_clipped - return_batch) ** 2
            qf_loss_max = torch.max(qf_loss_unclipped, qf_loss_clipped)
            qf_loss = 0.5 * qf_loss_max.mean()
        else:
            qf_loss = 0.5 * ((new_value - return_batch) ** 2).mean()

        return qf_loss

    def get_batch(self, advantages, returns) -> Tuple[torch.Tensor, ...]:
        b_obs = self.replay_buffer.states.reshape((-1,) + self.observation_space.shape)
        b_logprobs = self.replay_buffer.logprobs.reshape(-1)
        b_actions = self.replay_buffer.actions.reshape((-1,) + self.action_space.shape)
        b_values = self.replay_buffer.values.reshape((-1, self.num_rewards))
        b_advantages = advantages.reshape((-1, self.num_rewards))
        b_returns = returns.reshape((-1, self.num_rewards))
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values
