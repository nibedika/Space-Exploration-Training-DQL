import gym
from gym import spaces
import numpy as np
import pygame
import os

class SpaceExplorerEnv(gym.Env):
    """
    Custom environment for Space Explorer theme where the agent navigates through 
    a grid of planets to collect a single resource while avoiding asteroids and black holes.

    Observation space: Box space representing the grid environment.
    Action space: Discrete space representing Up, Down, Left, Right movements.
    Rewards: 
        -> +10 for collecting the resource.
        -> -20 for colliding with an asteroid.
        -> -50 for falling into a black hole.
        -> -5 for each step taken.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        super(SpaceExplorerEnv, self).__init__()

        # Define the dimensions of the grid
        self.grid_width = 10
        self.grid_height = 10

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.grid_width, self.grid_height), dtype=int)

        # Initialize the grid environment
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        # Define rewards for different events
        self.rewards = {
            'collect_resource': 10,
            'collision_asteroid': -20,
            'fall_into_black_hole': -50,
            'step_penalty': -5
        }

        # Define starting position
        self.starting_position = (0, 0)
        self.current_position = self.starting_position
        self.grid[self.starting_position] = 1  # Set starting position as a planet

        # Define static positions for obstacles and the single resource
        self.asteroids = [(3, 3), (3, 4), (4, 3)]
        self.black_holes = [(5, 5), (6, 6)]
        self.resource_position = (7, 7)  # Single resource position

        # Pygame window settings
        self.window_size = 500  # Size of the pygame window
        self.cell_size = self.window_size // self.grid_width  # Size of each cell in the grid

        # Load images for better visualization
        self.images = {
            'spaceship': pygame.image.load(os.path.join('Images/spaceship.png')),
            'asteroid': pygame.image.load(os.path.join('Images/asteroid.png')),
            'black_hole': pygame.image.load(os.path.join('Images/black_hole.png')),
            'resource': pygame.image.load(os.path.join('Images/resource.png')),
        }

        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Space Explorer")

        # Initialize step count for tracking
        self.step_count = 0

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        # Place the agent at the starting position
        self.current_position = self.starting_position
        self.grid[self.starting_position] = 1

        # Place asteroids, black holes, and the resource in their static positions
        for pos in self.asteroids:
            self.grid[pos] = 2  # Place an asteroid
        for pos in self.black_holes:
            self.grid[pos] = 3  # Place a black hole

        # Place the single resource
        self.grid[self.resource_position] = 4  # Place the resource

        self.step_count = 0  # Reset step count

        return self.grid

    def step(self, action):
        """
        Execute the action and return the new state, reward, and done flag.
        """
        self.step_count += 1
        reward = 0
        done = False

        # Determine the new position based on the action
        x, y = self.current_position
        if action == 0:  # Up
            new_position = (x - 1, y)
        elif action == 1:  # Down
            new_position = (x + 1, y)
        elif action == 2:  # Left
            new_position = (x, y - 1)
        elif action == 3:  # Right
            new_position = (x, y + 1)
        else:
            new_position = self.current_position

        # Check if the new position is valid
        if self._is_valid_position(new_position):
            # Update the agent's current position
            self.current_position = new_position

            # Apply the effects of the new position
            if self.grid[self.current_position] == 4:  # Collected the resource
                reward += self.rewards['collect_resource']
                done = True
            elif self.grid[self.current_position] == 2:  # Collided with an asteroid
                reward += self.rewards['collision_asteroid']
                done = True
            elif self.grid[self.current_position] == 3:  # Fell into a black hole
                reward += self.rewards['fall_into_black_hole']
                done = True

            # Update the grid with the new position of the agent
            self._update_grid()
        else:
            pass  # Invalid move

        # Apply step penalty
        reward += self.rewards['step_penalty']

        return self.grid, reward, done, {}

    def render(self):
        """
        Render the current state of the environment using pygame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((0, 0, 0))  # Fill the window with black for space background

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.grid[x, y] == 1:
                    self.window.blit(pygame.transform.scale(self.images['spaceship'], (self.cell_size, self.cell_size)), rect)
                elif self.grid[x, y] == 2:
                    self.window.blit(pygame.transform.scale(self.images['asteroid'], (self.cell_size, self.cell_size)), rect)
                elif self.grid[x, y] == 3:
                    self.window.blit(pygame.transform.scale(self.images['black_hole'], (self.cell_size, self.cell_size)), rect)
                elif self.grid[x, y] == 4:
                    self.window.blit(pygame.transform.scale(self.images['resource'], (self.cell_size, self.cell_size)), rect)
                pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # Draw grid lines

        pygame.display.flip()

    def close(self):
        """
        Clean up resources, if any.
        """
        pygame.quit()

    def _update_grid(self):
        """
        Helper function to update the grid with the agent's current position and other objects.
        """
        # Clear the grid of the agent's previous position
        self.grid = np.where(self.grid == 1, 0, self.grid)

        # Set the agent's new position
        self.grid[self.current_position] = 1

    def _is_valid_position(self, position):
        """
        Helper function to check if a position is valid within the grid.
        """
        x, y = position
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

# Register the environment with OpenAI Gym
gym.register('SpaceExplorer-v0', entry_point='padm_env:SpaceExplorerEnv')
