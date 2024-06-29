import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from methods.networks.utils import xavier_init


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_outputs=1, n_hidden=1):
        super(QNetwork, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 256),
            nn.ReLU(),
        )
        if n_hidden == 1:
            self.q.add_module("hidden_layer1", nn.Linear(256, 256))
            self.q.add_module("relu1", nn.ReLU())
        elif n_hidden == 2:
            self.q.add_module("hidden_layer1", nn.Linear(256, 512))
            self.q.add_module("relu1", nn.ReLU())
            self.q.add_module("hidden_layer2", nn.Linear(512, 256))
            self.q.add_module("relu2", nn.ReLU())
        elif n_hidden == 3:
            self.q.add_module("hidden_layer1", nn.Linear(256, 512))
            self.q.add_module("relu1", nn.ReLU())
            self.q.add_module("hidden_layer2", nn.Linear(512, 512))
            self.q.add_module("relu2", nn.ReLU())
            self.q.add_module("hidden_layer3", nn.Linear(512, 256))
            self.q.add_module("relu3", nn.ReLU())
        elif n_hidden == 4:
            self.q.add_module("hidden_layer1", nn.Linear(256, 512))
            self.q.add_module("relu1", nn.ReLU())
            self.q.add_module("hidden_layer2", nn.Linear(512, 1024))
            self.q.add_module("relu2", nn.ReLU())
            self.q.add_module("hidden_layer3", nn.Linear(1024, 512))
            self.q.add_module("relu3", nn.ReLU())
            self.q.add_module("hidden_layer4", nn.Linear(512, 256))
            self.q.add_module("relu4", nn.ReLU())
        else:
            raise ValueError("n_hidden must be between 1 and 4")

        self.q.add_module("output_layer", nn.Linear(256, num_outputs))

    def forward(self, state, action):
        xu = torch.cat([state.clone(), action.clone()], 1)
        x1 = self.q(xu)
        return x1


class DoubleQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_outputs=1, n_hidden=1):
        super(DoubleQNetwork, self).__init__()

        # Q1 architecture
        self.q1 = QNetwork(num_inputs, num_actions, num_outputs, n_hidden)

        # Q2 architecture
        self.q2 = QNetwork(num_inputs, num_actions, num_outputs, n_hidden)

    def forward(self, state, action):
        x1 = self.q1(state, action)
        x2 = self.q2(state, action)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        n_hidden=1,
        epsilon=1e-6,
        action_space=None,
    ):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon

        self.linears = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
        )
        if n_hidden == 1:
            self.linears.add_module("hidden_layer1", nn.Linear(256, 256))
            self.linears.add_module("relu1", nn.ReLU())
        elif n_hidden == 2:
            self.linears.add_module("hidden_layer1", nn.Linear(256, 512))
            self.linears.add_module("relu1", nn.ReLU())
            self.linears.add_module("hidden_layer2", nn.Linear(512, 256))
            self.linears.add_module("relu2", nn.ReLU())
        elif n_hidden == 3:
            self.linears.add_module("hidden_layer1", nn.Linear(256, 512))
            self.linears.add_module("relu1", nn.ReLU())
            self.linears.add_module("hidden_layer2", nn.Linear(512, 512))
            self.linears.add_module("relu2", nn.ReLU())
            self.linears.add_module("hidden_layer3", nn.Linear(512, 256))
            self.linears.add_module("relu3", nn.ReLU())
        elif n_hidden == 4:
            self.linears.add_module("hidden_layer1", nn.Linear(256, 512))
            self.linears.add_module("relu1", nn.ReLU())
            self.linears.add_module("hidden_layer2", nn.Linear(512, 1024))
            self.linears.add_module("relu2", nn.ReLU())
            self.linears.add_module("hidden_layer3", nn.Linear(1024, 512))
            self.linears.add_module("relu3", nn.ReLU())
            self.linears.add_module("hidden_layer4", nn.Linear(512, 256))
            self.linears.add_module("relu4", nn.ReLU())
        else:
            raise ValueError("n_hidden must be between 1 and 4")

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

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
        x = self.linears(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
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


class MLPPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        n_hidden=1,
    ):
        super(MLPPolicy, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
        )
        if n_hidden == 1:
            self.mlp.add_module("hidden_layer1", nn.Linear(256, 256))
            self.mlp.add_module("relu1", nn.ReLU())
        elif n_hidden == 2:
            self.mlp.add_module("hidden_layer1", nn.Linear(256, 512))
            self.mlp.add_module("relu1", nn.ReLU())
            self.mlp.add_module("hidden_layer2", nn.Linear(512, 256))
            self.mlp.add_module("relu2", nn.ReLU())
        elif n_hidden == 3:
            self.mlp.add_module("hidden_layer1", nn.Linear(256, 512))
            self.mlp.add_module("relu1", nn.ReLU())
            self.mlp.add_module("hidden_layer2", nn.Linear(512, 512))
            self.mlp.add_module("relu2", nn.ReLU())
            self.mlp.add_module("hidden_layer3", nn.Linear(512, 256))
            self.mlp.add_module("relu3", nn.ReLU())
        elif n_hidden == 4:
            self.mlp.add_module("hidden_layer1", nn.Linear(256, 512))
            self.mlp.add_module("relu1", nn.ReLU())
            self.mlp.add_module("hidden_layer2", nn.Linear(512, 1024))
            self.mlp.add_module("relu2", nn.ReLU())
            self.mlp.add_module("hidden_layer3", nn.Linear(1024, 512))
            self.mlp.add_module("relu3", nn.ReLU())
            self.mlp.add_module("hidden_layer4", nn.Linear(512, 256))
            self.mlp.add_module("relu4", nn.ReLU())
        else:
            raise ValueError("n_hidden must be between 1 and 4")

        self.mlp.add_module("output_layer", nn.Linear(256, num_actions))
        self.apply(xavier_init)

    def forward(self, state):
        action = self.mlp(state)
        return action

    def get_action(self, state):
        action = self.forward(state)
        return action.detach().cpu().numpy()
