import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, obs_size, action_size, hidden_size=128, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model(hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=50000)
        self.batch_size = 64

    def _build_model(self, hidden_size):
        return nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 🔧 Konwersja do jednego np.ndarray zamiast listy ndarrays
        states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

