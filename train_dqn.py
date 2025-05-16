import gymnasium as gym
import numpy as np
from solitaire_env import SpiderSolitaireEnv
from dqn_agent import DQNAgent

def preprocess_obs(obs_dict):
    columns = np.array(obs_dict["columns"], dtype=np.float32).flatten()
    lengths = np.array(obs_dict["column_lengths"], dtype=np.float32)
    deck_size = np.array([obs_dict["deck_size"]], dtype=np.float32)

    return np.concatenate([columns, lengths, deck_size])

class DQNTrainer:
    def __init__(self, env=None, episodes=500, max_steps=500):
        self.env = env if env is not None else SpiderSolitaireEnv()
        obs, _ = self.env.reset()
        obs_vec = preprocess_obs(obs)
        obs_size = obs_vec.shape[0]
        action_size = self.env.action_space.n
        self.agent = DQNAgent(obs_size=obs_size, action_size=action_size)
        self.episodes = episodes
        self.max_steps = max_steps

    def train(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            obs = preprocess_obs(obs)
            total_reward = 0

            done = False
            for step in range(self.max_steps):
                action = self.agent.act(obs)
                next_obs, reward, done, _, _ = self.env.step(action)
                next_obs = preprocess_obs(next_obs)

                self.agent.remember(obs, action, reward, next_obs, done)
                self.agent.replay()

                obs = next_obs
                total_reward += reward

                if done:
                    break

            print(f"Ep {ep+1}: Total reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
