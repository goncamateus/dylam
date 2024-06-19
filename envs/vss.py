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
            "reward_Weighted/Move": 0,
            "reward_Weighted/Ball": 0,
            "reward_Weighted/Energy": 0,
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
            "reward_Weighted/Move": 0,
            "reward_Weighted/Ball": 0,
            "reward_Weighted/Energy": 0,
            "Original_reward": 0,
        }
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.cumulative_reward_info

    def _calculate_reward_and_done(self):
        reward = np.zeros(4, dtype=np.float32)
        goal = False
        w_move = 0.248
        w_ball_grad = 0.25
        w_energy = 0.002
        w_goal = 0.5
        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] += 1
            self.cumulative_reward_info["reward_Goal_blue"] += 1
            self.cumulative_reward_info["Original_reward"] += 1 * w_goal
            reward[-1] = w_goal
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] -= 1
            self.cumulative_reward_info["reward_Goal_yellow"] += 1
            self.cumulative_reward_info["Original_reward"] += 1 * w_goal
            reward[-1] = -w_goal
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
                        w_move * move_reward,
                        w_ball_grad * grad_ball_potential,
                        w_energy * energy_penalty,
                    ]
                )

                self.cumulative_reward_info["reward_Move"] += move_reward
                self.cumulative_reward_info["reward_Ball"] += grad_ball_potential
                self.cumulative_reward_info["reward_Energy"] += energy_penalty
                self.cumulative_reward_info["reward_Weighted/Move"] += reward[0]
                self.cumulative_reward_info["reward_Weighted/Ball"] += reward[1]
                self.cumulative_reward_info["reward_Weighted/Energy"] += reward[2]
                self.cumulative_reward_info["Original_reward"] += (
                    w_move * move_reward
                    + w_ball_grad * grad_ball_potential
                    + w_energy * energy_penalty
                )

        return reward, goal

    def __ball_grad(self):
        """Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        """
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -np.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = np.sqrt(dx_d**2 + 2 * dy**2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            grad_ball_potential = ball_potential - self.previous_ball_potential

        self.previous_ball_potential = ball_potential

        return grad_ball_potential / 0.02

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
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        return move_reward / 1.2

    def __energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty / 92
