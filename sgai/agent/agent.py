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
        epsilon: float = 0,
        gamma: float = 0.9,
        input_size=11,
        hidden_size=256,
        output_size=3,
    ) -> None:
        self.number_of_games = 0
        self.epsilon = epsilon  # Controls randomness
        self.gamma = gamma  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # TODO Change to be dynamic based on IO complexity
        self.model = LinearQNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(
            self.model, learning_rate=LEARNING_RATE, gamma=self.gamma
        )

    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_left = GamePoint(head.x - 20, head.y)
    #     point_right = GamePoint(head.x + 20, head.y)
    #     point_up = GamePoint(head.x, head.y - 20)
    #     point_down = GamePoint(head.x, head.y + 20)

    #     left_direction = game.direction == Direction.LEFT
    #     right_direction = game.direction == Direction.RIGHT
    #     up_direction = game.direction == Direction.UP
    #     down_direction = game.direction == Direction.DOWN

    #     state = [
    #         # Danger straight
    #         (right_direction and game.is_collision(point_right))
    #         or (left_direction and game.is_collision(point_left))
    #         or (up_direction and game.is_collision(point_up))
    #         or (down_direction and game.is_collision(point_down)),
    #         # Danger right
    #         (up_direction and game.is_collision(point_right))
    #         or (down_direction and game.is_collision(point_left))
    #         or (left_direction and game.is_collision(point_up))
    #         or (right_direction and game.is_collision(point_down)),
    #         # Danger left
    #         (down_direction and game.is_collision(point_right))
    #         or (up_direction and game.is_collision(point_left))
    #         or (right_direction and game.is_collision(point_up))
    #         or (left_direction and game.is_collision(point_down)),
    #         # Move direction
    #         left_direction,
    #         right_direction,
    #         up_direction,
    #         down_direction,
    #         # Food location
    #         game.food.x < game.head.x,  # Food left
    #         game.food.x > game.head.x,  # Food right
    #         game.food.y < game.head.y,  # Food up
    #         game.food.y > game.head.y,  # Food down
    #     ]

    #     # debug(state)
    #     return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_state):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, game_state))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # List of tuples
            mini_sample: List[Tuple] = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample: List[Tuple] = self.memory

        states, actions, rewards, next_states, game_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_states)

    def train_short_memory(self, state, action, reward, next_state, game_state):
        self.trainer.train_step(state, action, reward, next_state, game_state)

    def get_action(self, state):
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


# def train():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
#     agent = Agent()
#     game = SnakeGame()
#     while True:
#         # Get old state
#         old_state = agent.get_state(game)

#         # Get Move based on the State
#         final_move = agent.get_action(old_state)

#         # Performe the Move and get the new State
#         reward, game_over, game_score = game.play_step(final_move)
#         new_state = agent.get_state(game)

#         # Train the short memory ON EACH MOVE
#         agent.train_short_memory(
#             state=old_state,
#             action=final_move,
#             reward=reward,
#             next_state=new_state,
#             game_state=game_over,
#         )

#         # Remember the state for long term training
#         agent.remember(
#             state=old_state,
#             action=final_move,
#             reward=reward,
#             next_state=new_state,
#             game_state=game_over,
#         )

#         if game_over:
#             # Train long memory, plot the result
#             game.reset_game_state()
#             agent.number_of_games += 1
#             agent.train_long_memory()

#             # Save the newest record
#             if game_score > record:
#                 record = game_score
#                 agent.model.save_model()

#          print(f"Game {agent.number_of_games}, Score {game_score}, Record {record}")

#             # Plot scores
#             plot_scores.append(game_score)
#             total_score += game_score
#             mean_score = total_score / agent.number_of_games
#             plot_mean_scores.append(mean_score)
#             plot(plot_scores, plot_mean_scores)
