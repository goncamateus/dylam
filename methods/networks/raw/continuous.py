import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from methods.networks.utils import xavier_init


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_outputs=1, hidden_dim=256):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

        self.apply(xavier_init)

    def forward(self, state, action):
        xu = torch.cat([state.clone(), action.clone()], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        epsilon=1e-6,
        action_space=None,
    ):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(xavier_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_action = torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob - log_action
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_action(self, state):
        action, _, _ = self.sample(state)
        return action.detach().cpu().numpy()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
