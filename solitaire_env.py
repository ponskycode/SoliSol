import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SpiderSolitaireEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Constants
        self.num_columns = 6
        self.num_ranks = 6
        self.num_suits = 1
        self.sets = 4
        self.max_seq_len = self.num_ranks  # e.g. 6
        self.action_space = spaces.Discrete(1 + self.num_columns * self.num_columns * self.max_seq_len)

        # Definition of observation and action spaces
        self.observation_space = spaces.Dict({
            "columns": spaces.Box(low=0, high=self.num_ranks, shape=(self.num_columns, 24), dtype=np.int32),
            "column_lengths": spaces.Box(low=0, high=24, shape=(self.num_columns,), dtype=np.int32),
            "deck_size": spaces.Discrete(25),  # 24 cards + 1 for the case of 0
        })

        # Action: (from_col, to_col) or drawing from the deck (special action)
        self.action_space = spaces.Discrete(self.num_columns * self.num_columns + 1)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Create and shuffle cards
        self.deck = [rank for _ in range(self.sets) for rank in range(self.num_ranks, 0, -1)]
        random.shuffle(self.deck)

        # Deal cards to columns
        self.columns = [[] for _ in range(self.num_columns)]
        for i in range(24 - self.num_columns * 4):  # e.g. leave 6*4 = 24 cards, 4 for each column
            self.columns[i % self.num_columns].append(self.deck.pop())

        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        padded_columns = np.zeros((self.num_columns, 24), dtype=np.int32)
        column_lengths = np.zeros(self.num_columns, dtype=np.int32)
        for i, col in enumerate(self.columns):
            padded_columns[i, :len(col)] = col
            column_lengths[i] = len(col)
        return {
            "columns": padded_columns,
            "column_lengths": column_lengths,
            "deck_size": len(self.deck)
        }

    def step(self, action):
        if action == 0:
            self._draw_from_deck()
            reward = -0.1
        else:
            action -= 1
            total = self.num_columns * self.max_seq_len
            from_col = action // total
            rest = action % total
            to_col = rest // self.max_seq_len
            seq_len = (rest % self.max_seq_len) + 1  # because we count from 0

            reward = self._move_sequence(from_col, to_col, seq_len)

        terminated = self._check_win()
        self.done = terminated
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _move_sequence(self, from_col, to_col, length):
        if from_col == to_col:
            return -1.0

        source = self.columns[from_col]
        target = self.columns[to_col]

        if len(source) < length:
            return -1.0  # not enough cards

        sequence = source[-length:]

        # check if this is a valid descending sequence
        for i in range(len(sequence) - 1):
            if sequence[i] != sequence[i + 1] + 1:
                return -1.0

        # check if can be placed on target
        if target:
            if sequence[0] != target[-1] - 1:
                return -1.0
        # if target is empty — allowed

        # perform the move
        del source[-length:]
        target.extend(sequence)

        # check if a full sequence was created
        if self._check_sequence(to_col):
            for _ in range(self.num_ranks):
                target.pop()
            return 1.0

        return 0.1

    def _check_sequence(self, col_idx):
        col = self.columns[col_idx]
        if len(col) < self.num_ranks:
            return False
        return col[-self.num_ranks:] == list(range(self.num_ranks, 0, -1))

    def _draw_from_deck(self):
        if not self.deck:
            return
        for col in self.columns:
            if self.deck:
                col.append(self.deck.pop())

    def _check_win(self):
        return all(len(col) == 0 for col in self.columns) and len(self.deck) == 0

def render_env(obs):
    print("\n====== GAME STATE ======")
    for i in range(len(obs['columns'])):
        col = obs['columns'][i][:obs['column_lengths'][i]]
        print(f"Column {i}: {list(col)}")
    print(f"Deck: {obs['deck_size']} cards")