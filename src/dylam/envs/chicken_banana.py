from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame import Surface
from pygame.time import Clock

ALPHABET_INT = {"x": 0, " ": 1, "s": 2, "b": 20, "c": 10, "g": 5}


class ChickenBanana(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, box_view=False):
        super().__init__()

        # Define the grid layout
        self.grid_layout = [
            ["x", "x", "x", "x", "x", "x", "G", "x", "x"],
            ["x", "x", "x", "x", "x", "x", " ", "x", "x"],
            ["x", "x", "x", "x", "x", "x", " ", "x", "x"],
            ["x", "x", "x", "x", "x", "x", " ", "x", "x"],
            ["C", " ", " ", " ", " ", " ", " ", " ", "B"],
            ["x", "x", "x", "x", "x", "x", " ", "x", "x"],
            ["x", "x", "x", "x", "x", "x", " ", "x", "x"],
            ["x", "x", "x", "x", "x", "x", "S", "x", "x"],
        ]
        self.box_view = box_view

        self.height = len(self.grid_layout)
        self.width = len(self.grid_layout[0])

        self.obs_map = self.__get_map_to_obs()

        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Define observation space
        self.n_states = sum(
            1 for row in self.grid_layout for cell in row if cell != "x"
        )
        self.observation_space = spaces.Discrete(self.n_states * 4)
        if box_view:
            self.observation_space = spaces.Box(
                low=np.zeros(4),
                high=np.array([12, 23, 1, 1]),
                shape=(4,),
                dtype=np.int32,
            )

        self.reward_dim = 3
        self.reward_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=float),
            high=np.array([100, 30, 70], dtype=float),
            shape=(self.reward_dim,),
            dtype=np.float32,
        )

        # Initialize state
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.agent_init_pos = np.zeros(2, dtype=np.int32)
        self.has_banana = 0
        self.has_chicken = 0
        self.banana_pos = np.zeros(2, dtype=np.int32)
        self.chicken_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.zeros(2, dtype=np.int32)

        # Find special positions
        self._find_special_positions()

        # Rendering setup
        self.render_mode = render_mode
        self.window: Optional[Surface] = None
        self.clock: Optional[Clock] = None
        self.cell_size = 40

        self.cumulative_reward_info = {
            "reward_Chicken": 0,
            "reward_Banana": 0,
            "reward_Objective": 0,
            "Original_reward": 0,
        }

    def __get_map_to_obs(self) -> dict:
        """Map grid cell position to observation index"""
        mapping = {}
        idx = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.grid_layout[y][x] != "x":
                    mapping[(x, y)] = idx
                    idx += 1
        return mapping

    def _find_special_positions(self):
        """Find positions of special items in the grid"""
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid_layout[y][x]
                if cell == "S":
                    self.agent_init_pos = np.array([x, y], dtype=np.int32)
                elif cell == "B":
                    self.banana_pos = (x, y)
                elif cell == "C":
                    self.chicken_pos = (x, y)
                elif cell == "G":
                    self.goal_pos = (x, y)

    def _get_obs(self):
        obs = self.obs_map[tuple(self.agent_pos)]
        if self.has_banana:
            obs += self.n_states
        if self.has_chicken:
            obs += 2 * self.n_states
        if self.box_view:
            obs = np.array(
                [
                    self.agent_pos[0],
                    self.agent_pos[1],
                    self.has_banana,
                    self.has_chicken,
                ],
                dtype=np.int32,
            )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset grid layout and agent position
        self.agent_pos = self.agent_init_pos.copy()

        # Reset collected items
        self.has_banana = 0
        self.has_chicken = 0

        observation = self._get_obs()
        self.cumulative_reward_info = {
            "reward_Chicken": 0,
            "reward_Banana": 0,
            "reward_Objective": 0,
            "Original_reward": 0,
        }

        if self.render_mode == "human":
            self._render_frame()
        self.step_count = 0
        return observation, self.cumulative_reward_info

    def step(self, action):
        rewards = np.zeros(3, dtype=float)

        # Save current position
        old_pos = self.agent_pos.copy()

        # Move agent based on action
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Right
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Down
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 3:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        # Check if new position is valid (not a wall)
        x, y = self.agent_pos
        if self.grid_layout[y][x] == "x":
            # Hit a wall, revert to old position
            self.agent_pos = old_pos
            terminated = False
        else:
            # Valid move
            cell_content = self.grid_layout[y][x]
            rewards[1] = (
                30 if cell_content == "B" and not self.has_banana else 0
            )  # Gasta 4 passos a mais
            rewards[2] = (
                70 if cell_content == "C" and not self.has_chicken else 0
            )  # Gasta 18 passos a mais
            # Check for item collection
            if (x, y) == self.banana_pos and not self.has_banana:
                self.has_banana = 1
            elif (x, y) == self.chicken_pos and not self.has_chicken:
                self.has_chicken = 1

            # Check if goal reached
            terminated = (x, y) == self.goal_pos

        observation = self._get_obs()

        self.cumulative_reward_info["reward_Banana"] += rewards[1]
        self.cumulative_reward_info["reward_Chicken"] += rewards[2]
        if terminated:
            rewards[0] = 100
            self.cumulative_reward_info["reward_Objective"] += 100
        self.cumulative_reward_info["Original_reward"] += rewards.sum()
        self.step_count += 1

        if self.render_mode == "human":
            self._render_frame()

        return (
            observation,
            rewards,
            terminated,
            self.step_count >= 80,
            self.cumulative_reward_info,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Chicken-Banana Environment")
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.width * self.cell_size, self.height * self.cell_size)
                )
            elif self.render_mode == "rgb_array":
                self.window = pygame.Surface(
                    (self.width * self.cell_size, self.height * self.cell_size)
                )
        if self.clock is None and self.render_mode in ["human", "rgb_array"]:
            self.clock = Clock()

        canvas = pygame.Surface(
            (self.width * self.cell_size, self.height * self.cell_size)
        )
        canvas.fill((255, 255, 255))

        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Draw cell background
                if self.grid_layout[y][x] == "x":
                    pygame.draw.rect(canvas, (100, 100, 100), rect)  # Walls - gray
                else:
                    pygame.draw.rect(
                        canvas, (240, 240, 240), rect
                    )  # Empty - light gray

                # Draw grid lines
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

                # Draw special items
                if (x, y) == self.goal_pos:
                    pygame.draw.rect(canvas, (0, 255, 0), rect)  # Goal - green
                    font = pygame.font.Font(None, 24)
                    text = font.render("G", True, (0, 0, 0))
                    canvas.blit(
                        text, (x * self.cell_size + 15, y * self.cell_size + 10)
                    )

                if (x, y) == self.banana_pos and not self.has_banana:
                    pygame.draw.rect(canvas, (255, 255, 0), rect)  # Banana - yellow
                    font = pygame.font.Font(None, 24)
                    text = font.render("B", True, (0, 0, 0))
                    canvas.blit(
                        text, (x * self.cell_size + 15, y * self.cell_size + 10)
                    )

                if (x, y) == self.chicken_pos and not self.has_chicken:
                    pygame.draw.rect(canvas, (255, 200, 100), rect)  # Chicken - orange
                    font = pygame.font.Font(None, 24)
                    text = font.render("C", True, (0, 0, 0))
                    canvas.blit(
                        text, (x * self.cell_size + 15, y * self.cell_size + 10)
                    )

        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size + 5,
            self.agent_pos[1] * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10,
        )
        pygame.draw.rect(canvas, (0, 0, 255), agent_rect)  # Agent - blue

        # Draw collected items indicator
        font = pygame.font.Font(None, 24)
        if self.has_banana:
            banana_text = font.render("Banana: ✓", True, (0, 0, 0))
            canvas.blit(banana_text, (10, 10))
        if self.has_chicken:
            chicken_text = font.render("Chicken: ✓", True, (0, 0, 0))
            canvas.blit(chicken_text, (10, 40))

        if self.render_mode == "human":
            if self.window:
                self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


# Example usage and test
if __name__ == "__main__":
    # Create and test the environment
    env = ChickenBanana(render_mode="human", box_view=False)
    print("Observation space:", env.observation_space)

    # Reset the environment
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)

    # Test a few steps
    for step in range(240):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Observation: {obs}")
        print(f"  Info: {info}")

        if terminated or truncated:
            print("Resetting environment.")
            obs, info = env.reset()

    env.close()
