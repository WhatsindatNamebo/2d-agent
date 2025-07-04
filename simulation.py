"""
Procedural 2D World with Emergent Agent Behavior
-----------------------------------------------
A simulation of AI agents in a procedurally generated 2D world.
Agents exhibit emergent behaviors based on their needs and environment.
"""

import pygame
import numpy as np
from noise import pnoise2
import random
import math
from typing import List, Tuple, Dict, Optional

# ===== GLOBAL CONSTANTS =====
# Display settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
CELL_SIZE = 5
FPS = 60

# Simulation settings
INITIAL_AGENTS = 50
WORLD_WIDTH = SCREEN_WIDTH // CELL_SIZE
WORLD_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class World:
    """Manages the procedurally generated 2D world and its biomes."""
    
    # Biome IDs
    DEEP_WATER = 0
    SHALLOW_WATER = 1
    BEACH = 2
    PLAINS = 3
    FOREST = 4
    MOUNTAIN = 5
    
    def __init__(self, width: int, height: int, seed: int):
        """Initialize the world with given dimensions and seed.
        
        Args:
            width: Width of the world in cells
            height: Height of the world in cells
            seed: Random seed for world generation
        """
        self.width = width
        self.height = height
        self.seed = seed
        self.grid = np.zeros((width, height), dtype=int)
        
        # Define biome colors
        self.biomes = {
            self.DEEP_WATER: (0, 0, 128),      # Dark Blue
            self.SHALLOW_WATER: (0, 0, 255),   # Blue
            self.BEACH: (210, 180, 140),       # Tan
            self.PLAINS: (34, 139, 34),        # Forest Green
            self.FOREST: (0, 100, 0),          # Dark Green
            self.MOUNTAIN: (139, 137, 137)     # Gray
        }
        
        # Generate the terrain
        self._generate_terrain()
    
    def _generate_terrain(self) -> None:
        """Generate the world terrain using Perlin noise."""
        # Scale affects the size of the noise features
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        for x in range(self.width):
            for y in range(self.height):
                # Generate noise value for this position
                nx = x/self.width - 0.5
                ny = y/self.height - 0.5
                
                # Generate Perlin noise value
                value = pnoise2(nx * scale, 
                              ny * scale,
                              octaves=octaves,
                              persistence=persistence,
                              lacunarity=lacunarity,
                              repeatx=1024,
                              repeaty=1024,
                              base=self.seed)
                
                # Map noise value to biome
                if value < -0.5:
                    self.grid[x][y] = self.DEEP_WATER
                elif value < -0.2:
                    self.grid[x][y] = self.SHALLOW_WATER
                elif value < 0.0:
                    self.grid[x][y] = self.BEACH
                elif value < 0.4:
                    self.grid[x][y] = self.PLAINS
                elif value < 0.7:
                    self.grid[x][y] = self.FOREST
                else:
                    self.grid[x][y] = self.MOUNTAIN
    
    def is_water(self, x: int, y: int) -> bool:
        """Check if a cell is water."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y] in (self.DEEP_WATER, self.SHALLOW_WATER)
        return False
    
    def is_land(self, x: int, y: int) -> bool:
        """Check if a cell is land (not water)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y] not in (self.DEEP_WATER, self.SHALLOW_WATER)
        return False
    
    def is_forest(self, x: int, y: int) -> bool:
        """Check if a cell is a forest."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y] == self.FOREST
        return False
    
    def get_nearest_water(self, x: int, y: int, radius: int = 50) -> Optional[Tuple[int, int]]:
        """Find the nearest water cell within the given radius.
        
        Args:
            x, y: Center position for the search
            radius: Search radius in cells (default: 50)
            
        Returns:
            Tuple of (x, y) coordinates of the nearest water cell, or None if none found
        """
        min_dist = float('inf')
        nearest = None
        
        # Check cells in a square around the position
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.is_water(nx, ny)):
                    dist = dx*dx + dy*dy  # No need for sqrt since we just compare
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (nx, ny)
        
        return nearest
    
    def get_nearest_forest(self, x: int, y: int, radius: int = 50) -> Optional[Tuple[int, int]]:
        """Find the nearest forest cell within the given radius.
        
        Args:
            x, y: Center position for the search
            radius: Search radius in cells (default: 50)
            
        Returns:
            Tuple of (x, y) coordinates of the nearest forest cell, or None if none found
        """
        min_dist = float('inf')
        nearest = None
        
        # Check cells in a square around the position
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.is_forest(nx, ny)):
                    dist = dx*dx + dy*dy  # No need for sqrt since we just compare
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (nx, ny)
        
        return nearest
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the world on the given surface."""
        for x in range(self.width):
            for y in range(self.height):
                color = self.biomes[self.grid[x][y]]
                pygame.draw.rect(screen, color, 
                               (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


class Agent:
    """Represents an autonomous agent in the simulation."""
    
    def __init__(self, x: int, y: int, world_size: Tuple[int, int]):
        """Initialize an agent at the given position.
        
        Args:
            x, y: Starting position
            world_size: Tuple of (width, height) of the world
        """
        self.x = x
        self.y = y
        self.world_width, self.world_height = world_size
        
        # Basic attributes
        self.energy = 100
        self.thirst = 100
        self.age = 0
        self.max_age = random.randint(500, 1000)
        self.state = 'wandering'
        self.target_pos = None  # Current target position (x, y)
        
        # Visual properties
        self.color = (255, 255, 0)  # Yellow
        self.size = CELL_SIZE - 1   # Slightly smaller than cell size
    
    def update(self, world: World) -> None:
        """Update the agent's state and position."""
        # Age the agent
        self.age += 1
        
        # Consume resources
        self.energy -= 0.5
        self.thirst -= 0.8
        
        # Store previous state to detect changes
        prev_state = self.state
        
        # Update state based on needs
        if self.thirst < 50:
            self.state = 'seeking_water'
        elif self.energy < 50:
            self.state = 'seeking_food'
        else:
            self.state = 'wandering'
        
        # Reset target if state changed
        if self.state != prev_state:
            self.target_pos = None
        
        # Check if we've reached our target
        if self.target_pos is not None:
            target_x, target_y = self.target_pos
            if (int(self.x) == target_x and int(self.y) == target_y):
                # Reached target - fulfill the need and clear target
                if self.state == 'seeking_water':
                    self.thirst = 100
                    self.state = 'wandering'
                    self.target_pos = None
                elif self.state == 'seeking_food':
                    self.energy = 100
                    self.state = 'wandering'
                    self.target_pos = None
        
        # Take action based on state
        if self.state == 'wandering':
            self._wander()
        elif self.state == 'seeking_water':
            self._seek_water(world)
        elif self.state == 'seeking_food':
            self._seek_food(world)
        
        # Ensure position is within bounds
        self.x = max(0, min(self.world_width - 1, self.x))
        self.y = max(0, min(self.world_height - 1, self.y))
    
    def _wander(self) -> None:
        """Move randomly."""
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        self.x = max(0, min(self.world_width - 1, self.x + dx))
        self.y = max(0, min(self.world_height - 1, self.y + dy))
    
    def _seek_water(self, world: World) -> None:
        """Move toward the nearest water source."""
        # Only find a new target if we don't have one
        if self.target_pos is None:
            self.target_pos = world.get_nearest_water(int(self.x), int(self.y))
            if self.target_pos is None:
                self._wander()  # Wander if no water found
                return
        
        # Move towards the target
        if self.target_pos:
            self._move_towards(self.target_pos[0], self.target_pos[1])
    
    def _seek_food(self, world: World) -> None:
        """Move toward the nearest forest."""
        # Only find a new target if we don't have one
        if self.target_pos is None:
            self.target_pos = world.get_nearest_forest(int(self.x), int(self.y))
            if self.target_pos is None:
                self._wander()  # Wander if no forest found
                return
        
        # Move towards the target
        if self.target_pos:
            self._move_towards(self.target_pos[0], self.target_pos[1])
    
    def _move_towards(self, target_x: int, target_y: int) -> None:
        """Move one step toward the target position.
        
        Args:
            target_x, target_y: The target position to move towards
        """
        dx = 0
        dy = 0
        
        # Calculate direction to target
        if target_x < self.x:
            dx = -1
        elif target_x > self.x:
            dx = 1
            
        if target_y < self.y:
            dy = -1
        elif target_y > self.y:
            dy = 1
        
        # Only move in one direction at a time (prevents diagonal movement)
        if dx != 0 and dy != 0:
            if random.random() < 0.5:
                dx = 0
            else:
                dy = 0
        
        # Update position with bounds checking
        self.x = max(0, min(self.world_width - 1, self.x + dx))
        self.y = max(0, min(self.world_height - 1, self.y + dy))
    
    def is_alive(self) -> bool:
        """Check if the agent is still alive."""
        return self.energy > 0 and self.age < self.max_age and self.thirst > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the agent on the screen."""
        # Change color based on state
        if self.state == 'seeking_water':
            color = (0, 191, 255)  # Deep Sky Blue
        elif self.state == 'seeking_food':
            color = (255, 165, 0)  # Orange
        else:
            color = self.color
            
        # Draw the agent
        pygame.draw.rect(screen, color, 
                        (self.x * CELL_SIZE, self.y * CELL_SIZE, 
                         self.size, self.size))


def main():
    """Main function to run the simulation."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Agent Simulation")
    clock = pygame.time.Clock()
    
    # Create world with a random seed
    seed = random.randint(0, 10000)
    print(f"World seed: {seed}")
    world = World(WORLD_WIDTH, WORLD_HEIGHT, seed)
    
    # Create initial agents on land
    agents = []
    for _ in range(INITIAL_AGENTS):
        while True:
            x = random.randint(0, WORLD_WIDTH - 1)
            y = random.randint(0, WORLD_HEIGHT - 1)
            if world.is_land(x, y):
                agents.append(Agent(x, y, (WORLD_WIDTH, WORLD_HEIGHT)))
                break
    
    # Set up font for UI
    font = pygame.font.SysFont('Arial', 16)
    
    # Main game loop
    running = True
    frame_count = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause
                    pass
        
        # Update agents
        for agent in agents[:]:
            agent.update(world)
            if not agent.is_alive():
                agents.remove(agent)
        
        # Occasionally spawn new agents
        if frame_count % 100 == 0 and len(agents) < 100:  # Cap at 100 agents
            x = random.randint(0, WORLD_WIDTH - 1)
            y = random.randint(0, WORLD_HEIGHT - 1)
            if world.is_land(x, y):
                agents.append(Agent(x, y, (WORLD_WIDTH, WORLD_HEIGHT)))
        
        # Draw everything
        world.draw(screen)
        
        # Draw agents
        for agent in agents:
            agent.draw(screen)
        
        # Draw UI
        population_text = f"Population: {len(agents)}"
        if agents:
            avg_energy = sum(agent.energy for agent in agents) / len(agents)
            energy_text = f"Avg Energy: {avg_energy:.1f}"
        else:
            energy_text = "Avg Energy: N/A"
            
        time_text = f"Time: {frame_count}"
        
        # Render text surfaces
        pop_surface = font.render(population_text, True, WHITE, BLACK)
        energy_surface = font.render(energy_text, True, WHITE, BLACK)
        time_surface = font.render(time_text, True, WHITE, BLACK)
        
        # Draw text backgrounds for better visibility
        pygame.draw.rect(screen, BLACK, (5, 5, 150, 60))
        
        # Blit text to screen
        screen.blit(pop_surface, (10, 10))
        screen.blit(energy_surface, (10, 30))
        screen.blit(time_surface, (10, 50))
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
        frame_count += 1
    
    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
