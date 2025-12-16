import torch
import torch.nn as nn
from gymnasium.spaces import Box

from dylam.methods.networks.architectures import DoubleQNetwork as BaseDoubleQNetwork
from dylam.methods.networks.architectures import MLPPolicy as BasePolicy
from dylam.methods.networks.architectures import QNetwork as BaseQNetwork


class FeaturesExtractor(nn.Module):
    def __init__(self, observation_space: Box, num_out_features: int = 128) -> None:
        super(FeaturesExtractor, self).__init__()
        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, num_out_features), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        return self.linear(self.cnn(observations))


class QNetwork(BaseQNetwork):
    def __init__(
        self,
        observation_space: Box,
        num_actions: int,
        num_out_features: int = 128,
        num_outputs: int = 1,
        num_hidden: int = 1,
    ):
        super().__init__(
            num_inputs=num_out_features,
            num_outputs=num_outputs,
            num_actions=num_actions,
            n_hidden=num_hidden,
        )
        self.features_extractor = FeaturesExtractor(
            observation_space,
            num_out_features,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor = torch.Tensor([]),
    ) -> torch.Tensor:
        features = self.features_extractor(observations)
        xu = torch.cat([features.clone(), actions.clone(), lambdas.clone()], 1)
        return self.q(xu)


class DoubleQNetwork(BaseDoubleQNetwork):
    def __init__(
        self,
        observation_space: Box,
        num_actions,
        num_out_features: int = 128,
        num_outputs=1,
        n_hidden=1,
    ):
        super().__init__(
            num_inputs=num_out_features,
            num_actions=num_actions,
            num_outputs=num_outputs,
            n_hidden=n_hidden,
        )
        self.features_extractor = FeaturesExtractor(
            observation_space,
            num_out_features=num_out_features,
        )

    def forward(self, state, action, lambdas=torch.Tensor([])):
        # In case of using GPI-LS, we need to pass the lambda values to the Q network
        features = self.features_extractor(state)
        lambdas = lambdas.to(state.device)
        x1 = self.q1(features, action, lambdas)
        x2 = self.q2(features, action, lambdas)
        return x1, x2


class Policy(BasePolicy):
    def __init__(
        self,
        observation_space: Box,
        num_actions: int,
        num_outputs: int = 128,
        n_hidden: int = 1,
    ):
        super(Policy, self).__init__(
            num_inputs=num_outputs,
            num_actions=num_actions,
            n_hidden=n_hidden,
        )
        self.features_extractor = FeaturesExtractor(
            observation_space,
            num_out_features=128,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(observations)
        logits = self.mlp(features)
        return logits
