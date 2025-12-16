import copy

import torch

from dylam.methods.networks.architectures import DoubleQNetwork, QNetwork
from dylam.methods.networks.mini_grid.architectures import (
    DoubleQNetwork as MiniGridDoubleQNetwork,
)
from dylam.methods.networks.mini_grid.architectures import (
    QNetwork as MiniGridQNetwork,
)


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class TargetCritic(TargetNet):
    def __call__(self, S, A, lambdas=torch.Tensor([])):
        lambdas = lambdas.to(S.device)
        output = (
            self.target_model(S, A, lambdas)
            if isinstance(self.model, QNetwork)
            or isinstance(self.model, DoubleQNetwork)
            or isinstance(self.model, MiniGridQNetwork)
            or isinstance(self.model, MiniGridDoubleQNetwork)
            else self.target_model(S)
        )
        return output


class TargetActor(TargetNet):
    def __call__(self, S):
        output = self.target_model(S)
        return output
