import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from snake_game import SnakeGame, Point
from constants import *


class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Right, 2:Down, 3:Left

        # Observation space: 11 values
        # [danger_straight, danger_right, danger_left,
        #  dir_left, dir_right, dir_up, dir_down,
        #  food_left, food_right, food_up, food_down]
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

    def _get_observation(self):
        head = self.game.head

        # Direction vectors
        DIRECTIONS = [Point(0, -BLOCK_SIZE), Point(BLOCK_SIZE, 0), Point(0, BLOCK_SIZE), Point(-BLOCK_SIZE, 0)]

        # Get current direction index
        try:
            dir_idx = DIRECTIONS.index(self.game.direction)
        except ValueError:
            # Fallback if direction is not in expected format
            dir_idx = 0

        # Calculate points for danger detection relative to current direction
        straight_point = Point(head.x + self.game.direction.x, head.y + self.game.direction.y)
        right_dir = DIRECTIONS[(dir_idx + 1) % 4]
        left_dir = DIRECTIONS[(dir_idx - 1 + 4) % 4]
        right_point = Point(head.x + right_dir.x, head.y + right_dir.y)
        left_point = Point(head.x + left_dir.x, head.y + left_dir.y)

        # Direction booleans
        dir_l = self.game.direction == Point(-BLOCK_SIZE, 0)
        dir_r = self.game.direction == Point(BLOCK_SIZE, 0)
        dir_u = self.game.direction == Point(0, -BLOCK_SIZE)
        dir_d = self.game.direction == Point(0, BLOCK_SIZE)

        state = [
            # Danger straight, right, left
            int(self.game._is_collision(straight_point)),
            int(self.game._is_collision(right_point)),
            int(self.game._is_collision(left_point)),

            # Current move direction (one-hot)
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),

            # Food location relative to head
            int(self.game.food.x < head.x),  # Food left
            int(self.game.food.x > head.x),  # Food right
            int(self.game.food.y < head.y),  # Food up
            int(self.game.food.y > head.y)  # Food down
        ]

        return np.array(state, dtype=np.float32)

    def step(self, action):
        # Convert discrete action to one-hot array
        action_array = np.zeros(4)
        action_array[action] = 1

        # Calculate distance to food before move for reward shaping
        current_dist_to_food = np.sqrt((self.game.head.x - self.game.food.x) ** 2 +
                                       (self.game.head.y - self.game.food.y) ** 2)

        # Execute game step
        reward, done, score = self.game.step(action_array)

        # Add distance-based reward shaping if game is not over
        if not done:
            new_dist_to_food = np.sqrt((self.game.head.x - self.game.food.x) ** 2 +
                                       (self.game.head.y - self.game.food.y) ** 2)

            if new_dist_to_food < current_dist_to_food:
                reward += REWARD_MOVED_CLOSER
            else:
                reward += REWARD_MOVED_AWAY

        obs = self._get_observation()
        info = {'score': score}

        # In gymnasium, step returns (obs, reward, terminated, truncated, info)
        # terminated = True when episode ends due to game rules (collision/death)
        # truncated = True when episode ends due to time limits (not applicable here)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)

        self.game.reset()
        observation = self._get_observation()
        info = {'score': 0}
        return observation, info

    def render(self, mode='human'):
        if mode == 'human':
            self.game._update_ui()
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    def close(self):
        pygame.quit()