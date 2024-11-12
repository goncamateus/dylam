import numpy as np

from envs.vss import VSSStratEnv
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Robot


class MAVSS(VSSStratEnv):

    def __init__(self, with_fault=False, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.with_fault = with_fault
        self.action_space = Box(low=-1, high=1, shape=(4,))
        self.reward_dim = 5 if with_fault else 4
        self.reward_space = Box(low=-1, high=1, shape=(self.reward_dim,))
        self.cumulative_reward_info = {
            "reward_Goal": 0,
            "reward_Agent1/Move": 0,
            "reward_Agent1/Ball": 0,
            "reward_Agent1/Energy": 0,
            "reward_Agent1/Efficiency": 0,
            "reward_Agent1/Fault": 0,
            "reward_Agent1/Collision": 0,
            "reward_Agent2/Move": 0,
            "reward_Agent2/Ball": 0,
            "reward_Agent2/Energy": 0,
            "reward_Agent2/Efficiency": 0,
            "reward_Agent2/Fault": 0,
            "reward_Agent2/Collision": 0,
            "reward_Goal_blue": 0,
            "reward_Goal_yellow": 0,
            "Original_reward": 0,
        }
        self.penalty_rect = np.array(
            [
                [0.6, 0.35],
                [0.75, -0.35],
            ]
        )

    def reset(self, *, seed=None, options=None):
        res = super().reset(seed=seed, options=options)
        self.cumulative_reward_info = {
            "reward_Goal": 0,
            "reward_Agent1/Move": 0,
            "reward_Agent1/Ball": 0,
            "reward_Agent1/Energy": 0,
            "reward_Agent1/Efficiency": 0,
            "reward_Agent1/Fault": 0,
            "reward_Agent1/Collision": 0,
            "reward_Agent2/Move": 0,
            "reward_Agent2/Ball": 0,
            "reward_Agent2/Energy": 0,
            "reward_Agent2/Efficiency": 0,
            "reward_Agent2/Fault": 0,
            "reward_Agent2/Collision": 0,
            "reward_Goal_blue": 0,
            "reward_Goal_yellow": 0,
            "Original_reward": 0,
        }
        return res

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions[:2]
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[:2])
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        self.actions[1] = actions[2:]
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[2:])
        commands.append(Robot(yellow=False, id=1, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(2, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(
                Robot(yellow=False, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue + i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(
                Robot(yellow=True, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )

        return commands

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
            grad_ball_potential = (
                ball_potential - self.previous_ball_potential
            ) / self.time_step

        self.previous_ball_potential = ball_potential

        return grad_ball_potential / 0.8

    def __efficiency_reward(self, ball_grad, energy):
        if np.isclose(ball_grad, 0, atol=1e-3):
            ball_grad = 0
        if np.isclose(energy, 0, atol=1e-3):
            reward_efficiency = 1
        else:
            reward_efficiency = ball_grad / energy
            reward_efficiency = reward_efficiency / 2
            reward_efficiency = np.clip(reward_efficiency, -1, 1)
        return reward_efficiency

    def __move_reward(self, robot_id=0):
        """Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array(
            [self.frame.robots_blue[robot_id].x, self.frame.robots_blue[robot_id].y]
        )
        robot_vel = np.array(
            [self.frame.robots_blue[robot_id].v_x, self.frame.robots_blue[robot_id].v_y]
        )
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        return move_reward / 1.2

    def __energy_penalty(self, robot_id=0):
        """Calculates the energy penalty"""

        en_penalty_1 = abs(self.sent_commands[robot_id].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[robot_id].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty / 92

    def _is_attack_fault(self):
        def is_in_penalty(x, y):
            return (
                self.penalty_rect[0][0] < x
                and self.penalty_rect[1][1] < y < self.penalty_rect[0][1]
            )

        def is_less_then_25_cm_diff(pos1, pos2):
            return np.linalg.norm(np.array(pos1) - np.array(pos2)) < 0.25

        pos_one = [self.frame.robots_blue[0].x, self.frame.robots_blue[0].y]
        pos_two = [self.frame.robots_blue[1].x, self.frame.robots_blue[1].y]
        one_in = is_in_penalty(self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
        two_in = is_in_penalty(self.frame.robots_blue[1].x, self.frame.robots_blue[1].y)
        ball_in = is_in_penalty(self.frame.ball.x, self.frame.ball.y)
        together = is_less_then_25_cm_diff(pos_one, pos_two)
        return one_in and two_in and ball_in and together

    def _collision_penalty(self):
        def is_less_then_20_cm_diff(pos1, pos2):
            return np.linalg.norm(np.array(pos1) - np.array(pos2)) < 0.2

        pos_one = [self.frame.robots_blue[0].x, self.frame.robots_blue[0].y]
        pos_two = [self.frame.robots_blue[1].x, self.frame.robots_blue[1].y]
        penalty = -int(is_less_then_20_cm_diff(pos_one, pos_two))
        return penalty

    def _calculate_reward_and_done(self):
        reward = np.zeros(self.reward_dim, dtype=np.float32)
        goal = False
        fault = False
        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] += 1
            self.cumulative_reward_info["reward_Goal_blue"] += 1
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.cumulative_reward_info["reward_Goal"] -= 1
            self.cumulative_reward_info["reward_Goal_yellow"] += 1
            goal = True
        elif self._is_attack_fault():
            fault = True
            self.cumulative_reward_info["reward_Agent1/Fault"] = 1
            self.cumulative_reward_info["reward_Agent2/Fault"] = 1
            if self.with_fault:
                goal = True
        if self.last_frame is not None:
            # Calculate ball potential
            grad_ball_potential = self.__ball_grad()
            efficiencies = []
            for i in [0, 1]:
                # Calculate Move ball
                move_reward = self.__move_reward(i)
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty(i)
                # Calculate efficiency reward
                efficiency_reward = self.__efficiency_reward(
                    move_reward, -energy_penalty
                )
                # Calculate collision penalty
                collision_penalty = self._collision_penalty()
                efficiencies.append(efficiency_reward)
                self.cumulative_reward_info[
                    f"reward_Agent{i+1}/Ball"
                ] += grad_ball_potential
                self.cumulative_reward_info[f"reward_Agent{i+1}/Move"] += move_reward
                self.cumulative_reward_info[
                    f"reward_Agent{i+1}/Energy"
                ] += energy_penalty
                self.cumulative_reward_info[
                    f"reward_Agent{i+1}/Efficiency"
                ] += efficiency_reward
                self.cumulative_reward_info[
                    f"reward_Agent{i+1}/Collision"
                ] += collision_penalty
            if self.with_fault:
                reward += np.array(
                    [
                        grad_ball_potential,
                        efficiencies[0],
                        efficiencies[1],
                        collision_penalty,
                        -1 if fault else 0,
                    ]
                )
            else:
                reward += np.array(
                    [
                        grad_ball_potential,
                        efficiencies[0],
                        efficiencies[1],
                        collision_penalty,
                    ]
                )
            self.cumulative_reward_info["Original_reward"] += np.sum(reward)

        return reward, goal
