#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # TODO CHENGE TO USE PATHLIB
    def save_model(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs((model_folder_path))

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model: nn.Module, learning_rate, gamma) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_state):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle multiple training sizes
        # If its not len 1 we have it in (n, x) shape
        if len(state.shape) == 1:
            # We have 1 number, we want it as (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_state = (game_state,)

        # 1: predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()
        for each_game_state in range(len(game_state)):
            Q_new = reward[each_game_state]
            if not game_state[each_game_state]:
                Q_new = reward[each_game_state] + self.gamma * torch.max(
                    self.model(next_state[each_game_state])
                )

            target[each_game_state][
                torch.argmax(action[each_game_state]).item()
            ] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if game_state
        # is not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
