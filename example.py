#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Tuple

import numpy as np
import torch

from sgai.agent.trainer import Trainer
from sgai.example.snake import Direction, GamePoint, SnakeGame


class MyTrainer(Trainer):
    def __init__(self, game, input_size: int, output_size: int):
        super().__init__(game, input_size, output_size)
        self.game = game

    def get_state(self) -> np.ndarray:
        head = self.game.snake[0]
        point_left = GamePoint(head.x - 20, head.y)
        point_right = GamePoint(head.x + 20, head.y)
        point_up = GamePoint(head.x, head.y - 20)
        point_down = GamePoint(head.x, head.y + 20)

        left_direction = self.game.direction == Direction.LEFT
        right_direction = self.game.direction == Direction.RIGHT
        up_direction = self.game.direction == Direction.UP
        down_direction = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (right_direction and self.game.is_collision(point_right))
            or (left_direction and self.game.is_collision(point_left))
            or (up_direction and self.game.is_collision(point_up))
            or (down_direction and self.game.is_collision(point_down)),
            # Danger right
            (up_direction and self.game.is_collision(point_right))
            or (down_direction and self.game.is_collision(point_left))
            or (left_direction and self.game.is_collision(point_up))
            or (right_direction and self.game.is_collision(point_down)),
            # Danger left
            (down_direction and self.game.is_collision(point_right))
            or (up_direction and self.game.is_collision(point_left))
            or (right_direction and self.game.is_collision(point_up))
            or (left_direction and self.game.is_collision(point_down)),
            # Move direction
            left_direction,
            right_direction,
            up_direction,
            down_direction,
            # Food location
            self.game.food.x < self.game.head.x,  # Food left
            self.game.food.x > self.game.head.x,  # Food right
            self.game.food.y < self.game.head.y,  # Food up
            self.game.food.y > self.game.head.y,  # Food down
        ]

        return np.array(state, dtype=int)

    def perform_action(self, state) -> Tuple[int, bool, int]:
        # Get random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


if __name__ == "__main__":
    mt = MyTrainer(game=SnakeGame(), input_size=11, output_size=3)
    mt.train(model_file="modeelllo.pth")
