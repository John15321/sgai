#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# from sgai.agent import train
from sgai.agent.agent import Agent
from sgai.agent.data_helper import plot


class Trainer(ABC):
    def __init__(
        self,
        game,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ):
        self.game = game
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def perform_action(self) -> Tuple[int, bool, int]:
        pass

    def train(self, model_file: str = "model.pth"):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )
        while True:
            # Get the old state
            old_state = self.get_state()

            # Get the move based on the State
            final_move = agent.get_action(old_state)

            # Perform the Move and get the new State
            reward, game_over, game_score = self.perform_action(final_move)
            # print(f"play step: {reward, game_over, game_score}")
            new_state = self.get_state()

            # Train the short memory ON EACH MOVE
            agent.train_short_memory(
                state=old_state,
                action=final_move,
                reward=reward,
                next_state=new_state,
                game_state=game_over,
            )

            # Remember the state for long term training
            agent.remember(
                state=old_state,
                action=final_move,
                reward=reward,
                next_state=new_state,
                game_state=game_over,
            )

            if game_over:
                # Train long memory, plot the result
                self.game.reset_game_state()
                agent.number_of_games += 1
                agent.train_long_memory()

                # Save the newest record
                if game_score > record:
                    record = game_score
                    agent.model.save_model(file_name=model_file)

                print(
                    f"Game {agent.number_of_games}, "
                    f"Score {game_score}, "
                    f"Record {record}"
                )

                # Plot the scores
                plot_scores.append(game_score)
                total_score += game_score
                mean_score = total_score / agent.number_of_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
