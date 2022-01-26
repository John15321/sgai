#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from collections import deque
from typing import List, Tuple

import torch

from .config import BATCH_SIZE, LEARNING_RATE, MAX_MEMORY
from .model import LinearQNet, QTrainer


def debug(msg):
    print(f"DEBUG: {msg}")


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        epsilon: float = 0,
        gamma: float = 0.9,
        hidden_size: int = 256,
    ) -> None:
        self.number_of_games = 0
        self.epsilon = epsilon  # Controls randomness
        self.gamma = gamma  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(
            self.model, learning_rate=LEARNING_RATE, gamma=self.gamma
        )

    def remember(self, state, action, reward, next_state, game_state):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, game_state))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample: List[Tuple] = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample: List[Tuple] = self.memory

        states, actions, rewards, next_states, game_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_states)

    def train_short_memory(self, state, action, reward, next_state, game_state):
        self.trainer.train_step(state, action, reward, next_state, game_state)

    def get_action(self, state):
        # Get random moves
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
