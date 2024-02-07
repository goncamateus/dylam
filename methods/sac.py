import copy
import os

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical
from torch.optim import Adam

from utils.buffer import ReplayBuffer, StratLastRewards


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def cnn_weights_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state.clone(), action.clone()], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2


class DiscreteQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_outputs=1, hidden_dim=256):
        super(QNetwork, self).__init__()

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

        self.apply(weights_init_)

    def forward(self, state):
        x1 = self.q1(state)
        x1 = x1.view(-1, self.num_actions, self.num_outputs)
        x2 = self.q2(state)
        x2 = x2.view(-1, self.num_actions, self.num_outputs)
        return x1, x2


class ImageQNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, num_outputs=1, **kwargs):
        super().__init__()
        self.num_actions = num_actions
        self.num_outputs = num_outputs
        self.conv1 = nn.Sequential(
            cnn_weights_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        self.conv2 = nn.Sequential(
            cnn_weights_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv1(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = cnn_weights_init(nn.Linear(output_dim, 512))
        self.fc2 = cnn_weights_init(nn.Linear(output_dim, 512))
        self.fc_q1 = cnn_weights_init(nn.Linear(512, num_actions * num_outputs))
        self.fc_q2 = cnn_weights_init(nn.Linear(512, num_actions * num_outputs))

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
        self.apply(weights_init_)
        self.fc_logits = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = self.encoder(state)
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs

    def sample(self, x):
        return self.get_action(x)


class ImageCategoricalPolicy(CategoricalPolicy):

    def __init__(
        self,
        obs_shape,
        num_actions,
        **kwargs,
    ):
        self.encoder = nn.Sequential(
            cnn_weights_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            output_dim = self.encoder(torch.zeros(1, *obs_shape)).shape[1]
        self.fc1 = cnn_weights_init(nn.Linear(output_dim, 512))
        self.fc_logits = cnn_weights_init(nn.Linear(512, num_actions))

    def forward(self, state):
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits


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

        self.apply(weights_init_)

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


class ImageGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        num_actions,
        log_sig_min=-5,
        log_sig_max=2,
        epsilon=1e-6,
        action_space=None,
        **kwargs,
    ):
        super(GaussianPolicy, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        self.encoder = nn.Sequential(
            cnn_weights_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            cnn_weights_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )
        with torch.inference_mode():
            output_dim = self.encoder(torch.zeros(1, *obs_shape)).shape[1]
        self.fc1 = cnn_weights_init(nn.Linear(output_dim, 512))
        self.mean_linear = cnn_weights_init(nn.Linear(512, num_actions))
        self.log_std_linear = cnn_weights_init(nn.Linear(512, num_actions))

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
        x = self.encoder(state)
        x = F.relu(self.fc1(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        return mean, log_std


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
    def __call__(self, S, A):
        return self.target_model(S, A)


class SAC(nn.Module):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
    ):
        super(SAC, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_inputs = np.array(observation_space.shape).prod()
        self.num_actions = np.array(action_space.shape).prod()
        self.reward_scaling = args.reward_scaling
        self.with_image = args.with_image
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.actor, self.critic = self.get_networks(
            action_space,
            args.epsilon,
            log_sig_min,
            log_sig_max,
            hidden_dim,
            self.with_image,
        )
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

        self.replay_buffer = self.get_replay_buffer(args.buffer_size)
        self.to(self.device)

    def get_networks(
        self,
        action_space,
        epsilon,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
        with_image=False,
    ):
        if with_image:
            actor = ImageGaussianPolicy(
                self.observation_space.shape,
                self.num_actions,
                log_sig_min=log_sig_min,
                log_sig_max=log_sig_max,
                hidden_dim=hidden_dim,
                epsilon=epsilon,
                action_space=action_space,
            )
            critic = ImageQNetwork(
                self.observation_space.shape,
                self.num_actions,
                hidden_dim=hidden_dim,
            )
        else:
            actor = GaussianPolicy(
                self.num_inputs,
                self.num_actions,
                log_sig_min=log_sig_min,
                log_sig_max=log_sig_max,
                hidden_dim=hidden_dim,
                epsilon=epsilon,
                action_space=action_space,
            )
            critic = QNetwork(
                self.num_inputs,
                self.num_actions,
                hidden_dim=hidden_dim,
            )
        return actor, critic

    def get_replay_buffer(self, buffer_size):
        return ReplayBuffer(buffer_size, self.device)

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.target_model.to(device)
        return super(SAC, self).to(device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.actor.get_action(state)

    def update_critic(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target.target_model(
                next_state_batch, next_state_action
            )

            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            min_qf_next_target[done_batch] = 0.0
            next_q_value = reward_batch + self.gamma * min_qf_next_target
        # Two Q-functions to mitigate
        # positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)

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

    def update_alpha(self, state_batch):
        alpha_loss = None
        if self.alpha_optim is not None:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state_batch)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach()
            alpha_loss = alpha_loss.mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        return alpha_loss

    def update_actor(self, state_batch):
        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch)

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
        qf1_loss, qf2_loss = self.update_critic(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch)

        return policy_loss, qf1_loss, qf2_loss, alpha_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.critic.state_dict(), path + "critic.pt")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.critic.load_state_dict(torch.load(path + "critic.pt"))
        self.actor.eval()
        self.critic.eval()


class SACStrat(SAC):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
        hidden_dim=256,
    ):
        self.is_dylam = args.dylam
        self.num_rewards = args.num_rewards
        super().__init__(
            args, observation_space, action_space, log_sig_min, log_sig_max, hidden_dim
        )
        if args.dylam:
            self.lambdas = (
                torch.ones(args.num_rewards).to(self.device) / args.num_rewards
            )
        else:
            self.lambdas = torch.Tensor(args.lambdas).to(self.device)
        self.r_max = torch.Tensor(args.r_max).to(self.device)
        self.r_min = torch.Tensor(args.r_min).to(self.device)
        self.rew_tau = args.dylam_tau
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def get_networks(
        self, action_space, epsilon, log_sig_min=-5, log_sig_max=2, hidden_dim=256
    ):
        actor = GaussianPolicy(
            self.num_inputs,
            self.num_actions,
            log_sig_min=log_sig_min,
            log_sig_max=log_sig_max,
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            action_space=action_space,
        )
        critic = QNetwork(
            self.num_inputs,
            self.num_actions,
            num_outputs=self.num_rewards,
            hidden_dim=hidden_dim,
        )
        return actor, critic

    def update_actor(self, state_batch):
        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi = (min_qf_pi * self.lambdas).sum(1).view(-1, 1)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch)

        return policy_loss, alpha_loss

    def update_lambdas(self):
        rew_mean_t = torch.Tensor(self.last_episode_rewards.mean()).to(self.device)
        if self.last_rew_mean is not None:
            rew_mean_t = rew_mean_t + (self.last_rew_mean - rew_mean_t) * self.rew_tau
        dQ = torch.clamp((self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1)
        expdQ = torch.exp(dQ) - 1
        self.lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
        self.last_rew_mean = rew_mean_t
