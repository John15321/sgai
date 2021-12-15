#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from enum import Enum
from typing import Tuple

import numpy as np
print("importing pygame")
import pygame
from ..example.config import GameColors

BLOCK_SIZE = 20
SPEED = 100

pygame.init()
print("initialized pygame")
font = pygame.font.Font("sgai/example/fonts/arial.ttf", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# class OutputLayerDirection(Enum):
#     STRAIGHT = np.array([1, 0, 0])
#     RIGHT_TURN = np.array([0, 1, 0])
#     LEFT_TURN = np.array([0, 0, 1])


class GamePoint:
    def __init__(self, x: int, y: int) -> None:
        self._x = None
        self._y = None

        self.x = x
        self.y = y

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @x.setter
    def x(self, x):
        self._x = x

    @y.setter
    def y(self, y):
        self._y = y


class SnakeGame:
    def __init__(self, w: int = 640, h: int = 480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset_game_state()

    def reset_game_state(self):
        # Initialize game state
        self.direction = Direction.RIGHT

        self.head = GamePoint(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            GamePoint(self.head.x - BLOCK_SIZE, self.head.y),
            GamePoint(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration_number = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = GamePoint(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action) -> Tuple[int, bool, int]:
        self.frame_iteration_number += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. Update the game state

        # Eat food: +10
        # Game Over: -10
        # Else: 0
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration_number > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # Check if it hits the boundary
        if (
            point.x > self.w - BLOCK_SIZE
            or point.x < 0
            or point.y > self.h - BLOCK_SIZE
            or point.y < 0
        ):
            return True
        # Check if it hits itself
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(GameColors.BLACK.value)

        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                GameColors.DARK_GREEN.value,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                GameColors.LIGHT_GREEN.value,
                pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
            )

        pygame.draw.rect(
            self.display,
            GameColors.RED.value,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(f"Score: {self.score}", True, GameColors.WHITE.value)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        move_index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[move_index]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (move_index + 1) % 4
            new_direction = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (move_index - 1) % 4
            new_direction = clock_wise[next_idx]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = GamePoint(x=x, y=y)
