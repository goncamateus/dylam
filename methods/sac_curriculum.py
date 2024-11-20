import torch

from methods.networks.architectures import GaussianPolicy, DoubleQNetwork
from methods.networks.targets import TargetCritic
from methods.sac import SACStrat
from torch.optim import lr_scheduler


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
        self.train_critic = args.train_critic
        self.scheduler_critic = lr_scheduler.LinearLR(
                self.critic_optim, start_factor=args.q_lr, total_iters=100000
            )
        self.scheduler_actor = None
        self.scheduler_alpha = None
        if args.load_actor:
            self.scheduler_actor = lr_scheduler.LinearLR(
                self.actor_optim, start_factor=args.policy_lr, total_iters=100000
            )
            self.scheduler_alpha = lr_scheduler.LinearLR(
                self.alpha_optim, start_factor=args.policy_lr, total_iters=100000
            )
        self.load_pretrained(args.model_path, load_actor=args.load_actor)

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

    def load_pretrained(self, model_path, load_actor=False):
        if model_path is not None:
            q_dict = torch.load(model_path + "/critic.pt", weights_only=True)
            self.critic.load_state_dict(q_dict)
            if not self.train_critic:
                self.critic.eval()
            self.critic_target = TargetCritic(self.critic)
            if load_actor:
                actor_dict = torch.load(model_path + "/actor.pt", weights_only=True)
                self.actor.load_state_dict(actor_dict)

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

    def update(self, batch_size, update_actor=False):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.replay_buffer.sample(batch_size)

        qf1_loss, qf2_loss = None, None
        if self.train_critic:
            reward_batch = reward_batch * self.reward_scaling
            qf1_loss, qf2_loss = self.update_critic(
                state_batch, action_batch, reward_batch, next_state_batch, done_batch
            )
            self.scheduler_critic.step()
        policy_loss = None
        alpha_loss = None
        if update_actor:
            policy_loss, alpha_loss = self.update_actor(state_batch)
        if self.scheduler_actor is not None:
            self.scheduler_actor.step()
            self.scheduler_alpha.step()
        return policy_loss, qf1_loss, qf2_loss, alpha_loss
