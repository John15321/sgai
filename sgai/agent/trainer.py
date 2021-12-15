#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC

import pygame

from sgai.agent import train
from sgai.example import SnakeGame

# from agent.game_agent import train
# from game.snake_game import SnakeGame


class Trainer(ABC):
    def get_state(self):
        pass

    def perform_action(self):
        pass

    pass


if __name__ == "__main__":
    train()
