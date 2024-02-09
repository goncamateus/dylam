import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from methods.networks.utils import xavier_init


class DiscreteQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_outputs=1, hidden_dim=256):
        super(DiscreteQNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_outputs),
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_outputs),
        )

        self.apply(xavier_init)

    def forward(self, state):
        x1 = self.q1(state)
        x1 = x1.view(-1, self.num_actions, self.num_outputs)
        x2 = self.q2(state)
        x2 = x2.view(-1, self.num_actions, self.num_outputs)
        return x1, x2


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim=256,
        action_space=None,
        **kwargs,
    ):
        super(CategoricalPolicy, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_logits = nn.Linear(hidden_dim, num_actions)
        self.apply(xavier_init)

    def forward(self, state):
        x = self.encoder(state)
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        action, _, _ = self.sample(x)
        return action.detach().cpu().numpy()

    def sample(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
