import pygame
import random
import numpy as np
from collections import namedtuple
from constants import *

Point = namedtuple('Point', 'x, y')


# Add arithmetic operations to Point
def point_add(p1, p2):
    return Point(p1.x + p2.x, p1.y + p2.y)


def point_sub(p1, p2):
    return Point(p1.x - p2.x, p1.y - p2.y)


def point_mul(p, scalar):
    return Point(p.x * scalar, p.y * scalar)


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(None, 25)
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('PPO Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game state."""
        self.direction = random.choice(
            [Point(0, -BLOCK_SIZE), Point(0, BLOCK_SIZE), Point(-BLOCK_SIZE, 0), Point(BLOCK_SIZE, 0)])
        self.head = Point(WIDTH // 2, HEIGHT // 2)

        # Create initial snake body using helper functions
        self.snake = [
            self.head,
            point_sub(self.head, self.direction),
            point_sub(self.head, point_mul(self.direction, 2))
        ]

        self.score = 0
        self.food = None
        self.obstacles = []
        self._place_food()
        self.game_over = False

    def _place_food(self):
        """Places food in a random location not occupied by the snake or obstacles."""
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def _update_obstacles(self):
        """Adds obstacles based on score. This is our difficulty scaling."""
        # Add a new obstacle for every 3 points scored
        num_obstacles_to_add = (self.score // 4) - len(self.obstacles)
        for _ in range(num_obstacles_to_add):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                new_obstacle = Point(x, y)
                if (new_obstacle not in self.snake and
                        new_obstacle != self.food and
                        new_obstacle not in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break
                attempts += 1

    def step(self, action):
        """
        Takes an action and updates the game state.
        Action is a one-hot encoded array: [Up, Right, Down, Left]
        Returns: reward, game_over, score
        """
        # Convert one-hot action to direction index
        action_idx = np.argmax(action)

        # Direction mapping: 0=Up, 1=Right, 2=Down, 3=Left
        DIRECTIONS = [Point(0, -BLOCK_SIZE), Point(BLOCK_SIZE, 0), Point(0, BLOCK_SIZE), Point(-BLOCK_SIZE, 0)]
        new_direction = DIRECTIONS[action_idx]

        # Prevent moving directly backwards (into itself)
        current_dir_idx = DIRECTIONS.index(self.direction)
        opposite_dir_idx = (current_dir_idx + 2) % 4

        if action_idx != opposite_dir_idx:
            self.direction = new_direction

        # Move snake
        self.head = point_add(self.head, self.direction)
        self.snake.insert(0, self.head)

        # Check for collisions
        reward = 0
        if self._is_collision():
            self.game_over = True
            reward = REWARD_GAME_OVER
            return reward, self.game_over, self.score

        # Check for food
        if self.head == self.food:
            self.score += 1
            reward = REWARD_EAT_FOOD
            self._place_food()
            self._update_obstacles()  # Increase difficulty
        else:
            self.snake.pop()

        return reward, self.game_over, self.score

    def _is_collision(self, pt=None):
        """Checks if a given point causes a collision."""
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > WIDTH - BLOCK_SIZE or pt.x < 0 or pt.y > HEIGHT - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        # Hits an obstacle
        if pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        """Draws everything on the screen."""
        self.display.fill((0, 0, 0))  # Black background

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 100, 255), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 0, 255), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.display, (128, 128, 128), pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()