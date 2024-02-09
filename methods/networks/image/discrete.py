import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from methods.networks.utils import kaiming_init


class ImageQNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, num_outputs=1, **kwargs):
        super().__init__()
        self.num_actions = num_actions
        self.num_outputs = num_outputs
        self.conv1 = nn.Sequential(
            kaiming_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        self.conv2 = nn.Sequential(
            kaiming_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv1(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = kaiming_init(nn.Linear(output_dim, 512))
        self.fc2 = kaiming_init(nn.Linear(output_dim, 512))
        self.fc_q1 = kaiming_init(nn.Linear(512, num_actions * num_outputs))
        self.fc_q2 = kaiming_init(nn.Linear(512, num_actions * num_outputs))

    def forward(self, x):
        x1 = F.relu(self.conv1(x / 255.0))
        x1 = F.relu(self.fc1(x1))
        q1 = self.fc_q1(x1)
        q1 = q1.view(-1, self.num_actions, self.num_outputs)

        x2 = F.relu(self.conv2(x / 255.0))
        x2 = F.relu(self.fc2(x2))
        q2 = self.fc_q2(x2)
        q2 = q2.view(-1, self.num_actions, self.num_outputs)

        return q1, q2


class ImageCategoricalPolicy(nn.Module):

    def __init__(
        self,
        obs_shape,
        num_actions,
        **kwargs,
    ):
        super(ImageCategoricalPolicy, self).__init__()
        self.encoder = nn.Sequential(
            kaiming_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            kaiming_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            output_dim = self.encoder(torch.zeros(1, *obs_shape)).shape[1]
        self.fc1 = kaiming_init(nn.Linear(output_dim, 512))
        self.fc_logits = kaiming_init(nn.Linear(512, num_actions))

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
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
