# Game_class.py
# DESCRIPTION:
# This script contains a simple interactive game called Click to Kill. The game involves 
# clicking on moving targets to score points. It is implemented using the pygame library.

# FEATURES
# Dynamic Target Movement: Targets move in random directions and bounce off walls.
# Interactive Gameplay: Players click on targets to "kill" them and earn points.
# Score Display: Continuously updates and displays the player's score during gameplay.
# Adjustable Difficulty: Target speed and movement patterns can be modified.
# How to Play:
# Run the script using Python 3 with pygame installed.
# Use the mouse to click on the moving targets.
# Earn points for each successful hit.
# The game ends when the user exits the window.
# Key Functions:
# spawn_target: Spawns a new target at a random location.
# move_targets: Updates target positions and handles wall collisions.
# check_click: Detects whether a click hits a target and removes it if so.
# run: Manages the game loop and handles rendering, input, and updates.

import pygame
import random
import sys
import os
import numpy as np
import scipy

class ClickToKillGame:
    def __init__(self, screen_width=1600, screen_height=1000, target_count=5):
        """
        Initialize the game.

        :param screen_width: Width of the game window
        :param screen_height: Height of the game window
        :param target_count: Number of targets to spawn
        """
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Click to Kill")

        # Load images
        self.target_image = pygame.image.load("target.png")  # Replace with your image
        self.target_image = pygame.transform.scale(self.target_image, (200, 200))

        self.background_color = (0, 0, 0)
        self.clock = pygame.time.Clock()
        self.targets = []
        self.target_count = target_count
        self.score = 0

        self.difficulty = np.int16(1)

        # Spawn targets
        for _ in range(target_count):
            self.spawn_target()

    def spawn_target(self):

        """Spawn a new target at a random position with a random speed."""
        x = random.randint(0, self.screen_width - 200)
        y = random.randint(0, self.screen_height - 200)
        speed_x = random.choice([-1, 1])*self.difficulty
        speed_y = random.choice([-1, 1])*self.difficulty
        self.targets.append({"rect": pygame.Rect(x, y, 200, 200), "speed": (speed_x, speed_y)})

    def move_targets(self):
        """Move all targets and bounce them off walls."""
        for target in self.targets:
            target["rect"].x += target["speed"][0]
            target["rect"].y += target["speed"][1]

            # Bounce off walls
            if target["rect"].left < 0 or target["rect"].right > self.screen_width:
                target["speed"] = (-target["speed"][0], target["speed"][1])
            if target["rect"].top < 0 or target["rect"].bottom > self.screen_height:
                target["speed"] = (target["speed"][0], -target["speed"][1])

    def check_click(self, pos):
        """Check if a click hits a target."""
        for target in self.targets[:]:
            if target["rect"].collidepoint(pos):
                self.targets.remove(target)
                self.score += 1
                self.spawn_target()  # Spawn a new target when one is killed
                break

    def run(self):
        """Run the game loop."""
        running = True
        while running:
            self.screen.fill(self.background_color)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.check_click(event.pos)

            # Move and draw targets
            self.move_targets()
            for target in self.targets:
                self.screen.blit(self.target_image, target["rect"].topleft)

            # Display score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))

            # Update display
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = ClickToKillGame()
    game.run()
