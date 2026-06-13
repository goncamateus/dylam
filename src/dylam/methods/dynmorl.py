import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from dylam.methods.networks.architectures import QNetwork
from dylam.methods.networks.targets import TargetCritic
from dylam.utils.buffer import ReplayWeightAwareBuffer
from dylam.utils.diverse_buffer import DiverseReplayBuffer


class DynMORL(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super(DynMORL, self).__init__()
        self.obs_size = np.array(observation_space.shape).prod()
        self.action_size = action_space.n
        self.num_rewards = args.num_rewards
        self.algorithm = args.algorithm
        self.gamma = args.gamma
        self.reward_scaling = args.reward_scaling
        self.lr = args.q_lr
        self.reset_on_w_change = args.reset_optimizer_on_w_change

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        self.q_network = QNetwork(
            num_inputs=self.obs_size + self.num_rewards,
            num_actions=0,
            num_outputs=self.action_size * self.num_rewards,
            n_hidden=args.n_hidden,
        )
        self.target_q_network = TargetCritic(self.q_network)
        self.optimizer = Adam(self.q_network.parameters(), lr=args.q_lr)

        self.epsilon = args.epsilon
        self.epsilon_decay_factor = args.epsilon_decay_factor
        self.epsilon_min = 0.05

        if args.memory_strategy == "der":
            self.replay_buffer = DiverseReplayBuffer(
                args.buffer_size, self.device, args.der_secondary_size
            )
        else:
            self.replay_buffer = ReplayWeightAwareBuffer(args.buffer_size, self.device)

        self.to(self.device)

    def to(self, device):
        self.q_network.to(device)
        self.target_q_network.target_model.to(device)
        return super(DynMORL, self).to(device)

    def _q_mat(self, network, observation, w):
        # returns shape (B, action_size, num_rewards)
        q_flat = network(observation, torch.Tensor([]).to(self.device), w)
        return q_flat.reshape(-1, self.action_size, self.num_rewards)

    def get_output(self, observation, w):
        q_mat = self._q_mat(self.q_network, observation, w)
        # scalarised: (B, |A|, num_rewards) @ (B, num_rewards, 1) -> (B, |A|)
        w_col = w.unsqueeze(-1)
        q_scalar = torch.bmm(q_mat, w_col).squeeze(-1)
        return torch.argmax(q_scalar, dim=1).cpu().numpy()

    def epsilon_greedy(self, observation, w):
        if np.random.random() < 1 - self.epsilon:
            action = self.get_output(observation, w)
        else:
            action = np.random.randint(
                low=0, high=self.action_size, size=(observation.shape[0],)
            )
        return action

    def epsilon_greedy_decay(self, observation, w):
        action = self.epsilon_greedy(observation, w)
        self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    def get_action(self, observation, w):
        with torch.no_grad():
            observation = torch.Tensor(observation).to(self.device)
            w_t = torch.Tensor(w).to(self.device)
            if w_t.ndim == 1:
                w_t = w_t.unsqueeze(0).expand(observation.shape[0], -1)
            action = self.epsilon_greedy_decay(observation, w_t)
        return action

    def _sample_weights(self, batch_size):
        raw = np.random.dirichlet([1.0] * self.num_rewards, size=batch_size).astype(
            np.float32
        )
        return torch.Tensor(raw).to(self.device)

    def update_q(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, w_stored):
        w_update = w_stored
        if self.algorithm == "cond":
            w_update = self._sample_weights(state_batch.shape[0])

        qf_losses = []
        q_mat = self._q_mat(self.q_network, state_batch, w_update)

        with torch.no_grad():
            tgt_mat = self._q_mat(
                self.target_q_network.target_model, next_state_batch, w_update
            )

        for j in range(self.num_rewards):
            with torch.no_grad():
                tgt_vals = tgt_mat[:, :, j].max(dim=1)[0]
                tgt_vals[done_batch] = 0.0
                targets = reward_batch[:, j] * self.reward_scaling + self.gamma * tgt_vals

            q_vals = q_mat[:, :, j].gather(1, action_batch).squeeze()
            qf_losses.append(F.mse_loss(q_vals, targets))

        total_loss = sum(qf_losses)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return [loss.item() for loss in qf_losses]

    def update(self, batch_size):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            w_stored,
        ) = self.replay_buffer.sample(batch_size)
        action_batch = action_batch.long()
        return self.update_q(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, w_stored
        )

    def reset_optimizer(self):
        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_network.state_dict(), path + "q_network.pt")

    def load(self, path):
        self.q_network.load_state_dict(
            torch.load(
                path + "q_network.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.q_network.eval()
