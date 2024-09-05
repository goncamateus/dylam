import os
import numpy as np

from utils.buffer import StratLastRewards


class QLearning:
    def __init__(self, args, observation_space, action_space):
        # alpha=1e-1, gamma=0.99, strategy=0
        self.obs_size = observation_space.n
        self.action_size = action_space.n
        self.q_table = np.zeros((self.obs_size, self.action_size))
        self.alpha = args.q_lr
        self.gamma = args.gamma
        self.strategy = args.strategy
        self.reward_scaling = args.reward_scaling

        self.epsilon = 0.15 if args.strategy == 0 else 0.8
        self.epsilon_decay_factor = (
            args.epsilon_decay_factor if args.strategy == 1 else 0
        )
        self.epsilon_min = 0.05

        self.softmax_temperature = args.softmax_temperature if args.strategy == 2 else 0

        self.n_counter = np.zeros((self.obs_size, self.action_size))
        self.total_count = 0

    def get_output(self, observation):
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def get_action(self, observation):
        if self.strategy == 0:
            action = self.epsilon_greedy(observation)
        elif self.strategy == 1:
            action = self.epsilon_greedy_decay(observation)
        elif self.strategy == 2:
            action = self.softmax(observation)
        elif self.strategy == 3:
            action = self.ucb(observation)
        return action

    def epsilon_greedy(self, observation):
        if np.random.random() < 1 - self.epsilon:
            action = self.get_output(observation)
        else:
            action = np.random.randint(self.action_size)
        return action

    def epsilon_greedy_decay(self, observation):
        action = self.epsilon_greedy(observation)
        self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return action

    def softmax(self, observation):
        temperature = self.softmax_temperature
        logits = self.q_table[observation] / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        action = np.random.choice(self.action_size, p=probs)
        return action

    def ucb(self, observation):
        n = self.n_counter[observation].sum()
        c = 1
        if n == 0:
            action = np.random.randint(self.action_size)
            return action
        ucb = self.q_table[observation] + c * np.sqrt(np.log(self.total_count) / n)
        action = np.argmax(ucb)
        self.n_counter[observation][action] += 1
        return action

    def update_policy(self, observation, action, reward, next_obs):
        reward = reward * self.reward_scaling

        update = reward + self.gamma * (
            self.q_table[next_obs].max() - self.q_table[observation][action]
        )
        self.q_table[observation][action] = (
            self.q_table[observation][action] + self.alpha * update
        )
        self.total_count += 1

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(path + "q_table.npy", self.q_table)

    def load(self, path):
        self.q_table = np.load(path + "q_table.npy")


class drQ(QLearning):

    def __init__(self, args, observation_space, action_space):
        super().__init__(args, observation_space, action_space)
        self.num_rewards = args.num_rewards
        self.components_q = np.zeros((self.num_rewards, *self.q_table.shape))
        self.lambdas = np.ones(self.num_rewards) / self.num_rewards

    def update_component_tables(self, observation, action, reward, next_obs):
        def get_bootstrap_q(i):
            q_value = (
                self.components_q[i][next_obs].max()
                - self.components_q[i][observation][action]
            )
            return q_value

        for i in range(self.num_rewards):
            update = reward[i] + self.gamma * get_bootstrap_q(i)
            self.components_q[i][observation][action] = (
                self.components_q[i][observation][action] + self.alpha * update
            )

    def update_policy(self, observation, action, reward, next_obs):
        reward = reward * self.reward_scaling
        self.update_component_tables(observation, action, reward, next_obs)
        Qs = 0
        for i in range(self.num_rewards):
            Qs += self.lambdas[i] * self.components_q[i][observation][action]
        self.q_table[observation][action] = Qs

    def save(self, path):
        super().save(path)
        np.save(path + "components_q.npy", self.components_q)

    def load(self, path):
        super().load(path)
        self.components_q = np.load(path + "components_q.npy")


class QDyLam(drQ):

    def __init__(self, args, observation_space, action_space):
        super().__init__(args, observation_space, action_space)
        self.r_max = np.array(args.r_max)
        self.r_min = np.array(args.r_min)
        self.rew_tau = args.dylam_tau
        self.episode_reward = np.zeros(args.num_rewards)
        self.last_reward_mean = None
        self.last_episode_rewards = StratLastRewards(args.dylam_rb, self.num_rewards)

    def get_output(self, observation):
        lambdas = self.lambdas.reshape(-1, 1)
        q_value_stratified = self.components_q[:, observation]
        q_value = (q_value_stratified * lambdas).sum(axis=0)
        if np.all(q_value == q_value[0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_value)
        return action

    def softmax(self, observation):
        temperature = self.softmax_temperature
        lambdas = self.lambdas.reshape(-1, 1)
        q_value_stratified = self.components_q[:, observation]
        q_value = (q_value_stratified * lambdas).sum(axis=0)
        logits = q_value / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        action = np.random.choice(self.action_size, p=probs)
        return action

    def ucb(self, observation):
        n = self.n_counter[observation].sum()
        c = 1
        if n == 0:
            action = np.random.randint(self.action_size)
            return action
        lambdas = self.lambdas.reshape(-1, 1)
        q_value_stratified = self.components_q[:, observation]
        q_value = (q_value_stratified * lambdas).sum(axis=0)
        ucb = q_value + c * np.sqrt(np.log(self.total_count) / n)
        action = np.argmax(ucb)
        self.n_counter[observation][action] += 1
        return action

    def update_policy(self, observation, action, reward, next_obs):
        reward = reward * self.reward_scaling
        self.update_component_tables(observation, action, reward, next_obs)

    def add_episode_reward(self, reward, termination, truncation):
        if self.num_rewards == 1:
            reward = reward.reshape(-1, 1)
        self.episode_reward += reward
        if termination or truncation:
            self.last_episode_rewards.add(self.episode_reward)
            self.episode_reward = np.zeros(self.num_rewards)

    def update_lambdas(self):
        if self.last_episode_rewards.can_do():
            rew_mean_t = self.last_episode_rewards.mean()
            if self.last_reward_mean is not None:
                rew_mean_t = (
                    rew_mean_t + (self.last_reward_mean - rew_mean_t) * self.rew_tau
                )
            dQ = np.clip((self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1)
            expdQ = np.exp(dQ) - 1
            self.lambdas = expdQ / (np.sum(expdQ, 0) + 1e-4)
            self.last_reward_mean = rew_mean_t
