import numpy as np

from gymnasium.spaces import Box
from rsoccer_gym.vss.env_vss.vss_gym import VSSEnv


class VSSStratEnv(VSSEnv):

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.reward_dim = 4
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Goal": 0,
            "reward_Move": 0,
            "reward_Energy": 0,
            "reward_Ball": 0,
            "reward_Goal_blue": 0,
            "reward_Goal_yellow": 0,
            "Original_reward": 0,
        }

    def reset(self, *, seed=None, options=None):
        self.cumulative_reward_info = {
            "reward_Goal": 0,
            "reward_Move": 0,
            "reward_Energy": 0,
            "reward_Ball": 0,
            "reward_Goal_blue": 0,
            "reward_Goal_yellow": 0,
            "Original_reward": 0,
        }
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.cumulative_reward_info

    def _calculate_reward_and_done(self):
        reward = np.zeros(4, dtype=np.float32)
        goal = False
        w_move = 0.018
        w_ball_grad = 0.068
        w_energy = 0.002
        w_goal = 0.911
        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] += 1
            self.cumulative_reward_info["reward_Goal_blue"] += 1
            self.cumulative_reward_info["Original_reward"] += 1 * w_goal
            reward[-1] = 1
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] -= 1
            self.cumulative_reward_info["reward_Goal_yellow"] += 1
            self.cumulative_reward_info["Original_reward"] += 1 * w_goal
            reward[-1] = -1
            goal = True
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward[:-1] += np.array(
                    [
                        move_reward,
                        grad_ball_potential,
                        energy_penalty,
                    ]
                )

                self.cumulative_reward_info["reward_Move"] += move_reward
                self.cumulative_reward_info["reward_Ball"] += grad_ball_potential
                self.cumulative_reward_info["reward_Energy"] += energy_penalty
                self.cumulative_reward_info["Original_reward"] += (
                    w_move * move_reward
                    + w_ball_grad * grad_ball_potential
                    + w_energy * energy_penalty
                )

        return reward, goal

    def __ball_grad(self):
        assert self.last_frame is not None

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        goal_pos = np.array([self.field.length / 2, 0])
        last_ball_dist = np.linalg.norm(goal_pos - last_ball_pos)

        # Calculate new ball dist
        ball = self.frame.ball
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal_pos - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist

        return ball_dist_rw / 0.03

    def __move_reward(self):
        """Calculate Move to ball reward
        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        vel_norm = np.linalg.norm(robot_vel)
        vel_norm = vel_norm if not np.isclose(vel_norm, 0, atol=1e-2) else 0

        robot_ball = ball - robot
        vec_norm = np.linalg.norm(robot_ball)
        vec_norm = vec_norm if not np.isclose(vec_norm, 0, atol=1e-2) else 0

        move_reward = np.dot(robot_ball, robot_vel)
        magnitudes = vel_norm * vec_norm
        move_reward = move_reward / magnitudes if magnitudes > 0 else 0
        return move_reward

    def __energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2) / 92.15338
        return energy_penalty
