#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame

# from agent.game_agent import train
# from game.snake_game import SnakeGame

from sgai.agent import train
from sgai.example import SnakeGame

from abc import ABC

class Trainer(ABC):


    def get_state(self):
        pass

    def perform_action(self):
        pass

    pass



if __name__ == "__main__":
    train()