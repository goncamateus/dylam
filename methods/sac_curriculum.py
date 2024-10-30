import torch

from methods.networks.architectures import GaussianPolicy, DoubleQNetwork
from methods.networks.targets import TargetCritic
from methods.sac import SACStrat


class SACCur(SACStrat):
    def __init__(
        self,
        args,
        observation_space,
        action_space,
        log_sig_min=-5,
        log_sig_max=2,
    ):
        self.considered_indices = args.considered_indices
        self.ori_num_rewards = args.ori_num_rewards
        super().__init__(
            args, observation_space, action_space, log_sig_min, log_sig_max
        )
        self.load_pretrained(args.q_path)

    def get_networks(self):
        actor = GaussianPolicy(
            self.num_inputs,
            self.num_actions,
            log_sig_min=self.log_sig_min,
            log_sig_max=self.log_sig_max,
            n_hidden=1,
            epsilon=self.epsilon,
            action_space=self.action_space,
        )
        critic = DoubleQNetwork(
            self.num_inputs,
            self.num_actions,
            num_outputs=self.ori_num_rewards,
            n_hidden=self.n_hidden,
        )
        return actor, critic

    def load_pretrained(self, q_path):
        if q_path is not None:
            q_dict = torch.load(q_path, weights_only=True)
            self.critic.load_state_dict(q_dict)
            self.critic_target = TargetCritic(self.critic)

    def update_actor(self, state_batch):
        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)[:, self.considered_indices]
        min_qf_pi = torch.einsum("ij,j->i", min_qf_pi, self.lambdas).view(-1, 1)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi
        policy_loss = policy_loss - min_qf_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        alpha_loss = self.update_alpha(state_batch)

        return policy_loss, alpha_loss
