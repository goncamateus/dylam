import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from methods.networks.architectures import QNetwork
from methods.networks.targets import TargetCritic

from utils.buffer import ReplayBuffer, StratLastRewards


class DQN(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super(DQN, self).__init__()
        self.obs_size = np.array(observation_space.shape).prod()
        self.action_size = action_space.n

        self.gamma = args.gamma
        self.n_hidden = args.n_hidden
        self.num_rewards = args.num_rewards

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        self.q_network = QNetwork(
            self.obs_size,
            num_actions=0,
            num_outputs=self.action_size,
            n_hidden=args.n_hidden,
        )
        self.target_q_network = TargetCritic(self.q_network)
        self.optimizer = Adam(self.q_network.parameters(), lr=args.q_lr)

        self.epsilon = 0.8
        self.epsilon_decay_factor = args.epsilon_decay_factor
        self.epsilon_min = 0.05

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.device)
        self.to(self.device)

    def to(self, device):
        self.q_network.to(device)
        self.target_q_network.target_model.to(device)
        return super(DQN, self).to(device)

    def get_output(self, observation):
        _ = torch.Tensor([]).to(self.device)
        q_values = self.q_network(observation, _)
        action = torch.argmax(q_values, dim=1).cpu().numpy()
        return action

    def epsilon_greedy(self, observation):
        if np.random.random() < 1 - self.epsilon:
            action = self.get_output(observation)
        else:
            action = np.random.randint(self.action_size)
        return action

    def epsilon_greedy_decay(self, observation):
        action = self.epsilon_greedy(observation)
        self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    def get_action(self, observation):
        with torch.no_grad():
            observation = torch.Tensor(observation).to(self.device)
            action = self.epsilon_greedy_decay(observation)
        return action

    def update_q(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
    ):
        _ = torch.Tensor([]).to(self.device)
        with torch.no_grad():
            target_q_values = self.target_q_network(next_state_batch, _).max(dim=1)
            target_q_values[done_batch] = 0
            next_q_values = reward_batch + self.gamma * target_q_values
        q_values = self.q_network(state_batch, _)
        q_values = q_values.gather(1, action_batch).squeeze(1)
        qf_loss = F.mse_loss(q_values, next_q_values)

        self.optimizer.zero_grad()
        qf_loss.backward()
        self.optimizer.step()

        return qf_loss.item()

    def update(self, batch_size):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)
        reward_batch = reward_batch * self.reward_scaling
        qf_loss = self.update_q(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        return qf_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_network.state_dict(), path + "q_network.pt")

    def load(self, path):
        self.q_network.load_state_dict(
            torch.load(path + "q_network.pt", map_location=self.device)
        )
        self.q_network.eval()


class DQNStrat(DQN):

    def __init__(self, args, observation_space, action_space):
        super(DQNStrat, self).__init__(args, observation_space, action_space)

        def q_net():
            return QNetwork(
                self.obs_size,
                num_actions=0,
                num_outputs=self.action_size,
                n_hidden=args.n_hidden,
            )

        self.q_networks = nn.ModuleList([q_net() * self.num_rewards])
        self.target_q_networks = nn.ModuleList(
            [TargetCritic(net) for net in self.q_networks]
        )
        self.optimizer = Adam(self.q_networks.parameters(), lr=args.q_lr)

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

    def get_q_value_from_components(self, observation):
        _ = torch.Tensor([]).to(self.device)
        components_values = torch.zeros(
            self.num_rewards, observation.shape[0], self.action_size
        ).to(self.device)
        for i in range(self.num_rewards):
            q_values = self.q_networks[i](observation, _)
            components_values[i] = q_values * self.lambdas[i]
        q_values = components_values.sum(dim=0)
        return q_values

    def get_output(self, observation):
        q_values = self.get_q_value_from_components(observation)
        action = torch.argmax(q_values, dim=1).cpu().numpy()
        return action

    def update_q(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
    ):
        _ = torch.Tensor([]).to(self.device)
        qf_losses = []
        for i, (q_net, target_q_net) in enumerate(
            zip(self.q_networks, self.target_q_networks)
        ):
            with torch.no_grad():
                target_q_values = target_q_net(next_state_batch, _).max(dim=1)
                target_q_values[done_batch] = 0
                next_q_values = reward_batch[:, i] + self.gamma * target_q_values
            q_values = q_net(state_batch, _)
            q_values = q_values.gather(1, action_batch).squeeze(1)
            qf_loss = F.mse_loss(q_values, next_q_values)
            qf_losses.append(qf_loss)

        self.optimizer.zero_grad()
        for loss in qf_losses:
            loss.backward()
        self.optimizer.step()

        return [loss.item() for loss in qf_losses]

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
