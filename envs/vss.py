import gymnasium as gym
import numpy as np

from rsoccer_gym.vss.env_vss import VSSEnv


class VSS(VSSEnv):

    def __init__(self, *args, **kwargs):
        super(VSS, self).__init__(*args, **kwargs)
        self.reward_dim = 4
        self.reward_space = gym.spaces.Box(
            low=np.array([-1, -1, -2, -1]),
            high=np.array([1, 1, 0, 1]),
            shape=(4,),
            dtype=float,
        )
        self.cumulative_reward_info = {
            "reward_ball": 0,
            "reward_goal": 0,
            "reward_move": 0,
            "reward_energy": 0,
            "Original_reward": 0,
        }

    def reset(self, *args, **kwargs):
        self.cumulative_reward_info = {
            "reward_ball": 0,
            "reward_goal": 0,
            "reward_move": 0,
            "reward_energy": 0,
            "Original_reward": 0,
        }
        return super().reset(*args, **kwargs)

    def _calculate_reward_and_done(self):
        reward = np.zeros(self.reward_dim)
        goal = False

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            reward[3] = 10
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            reward[3] = -10
            goal = True
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                ball_grad_reward = self.__ball_grad()
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward[0] = move_reward*2.5*0.2
                reward[1] = ball_grad_reward*3*0.8
                reward[2] = energy_penalty*2e-4
                self.cumulative_reward_info["reward_move"] += reward[0]
                self.cumulative_reward_info["reward_ball"] += reward[1]
                self.cumulative_reward_info["reward_energy"] += reward[2]

        self.cumulative_reward_info["reward_goal"] += reward[3]/10
        self.cumulative_reward_info["Original_reward"] += reward.sum()
        return reward, goal

    def step(self, action):
        observation, reward, termination, truncated, _ = super().step(action)
        return observation, reward, termination, truncated, self.cumulative_reward_info

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
        if self.previous_ball_potential is not None:
            grad_ball_potential = ball_potential - self.previous_ball_potential
            grad_ball_potential /= self.max_v

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

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

        return move_reward

    def __energy_penalty(self):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.actions[0][0])
        en_penalty_2 = abs(self.actions[0][1])
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty
