#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame

# from agent.game_agent import train
# from game.snake_game import SnakeGame

from sgai.agent.agent import train
from sgai.example.snake import SnakeGame

if __name__ == "__main__":
    train()