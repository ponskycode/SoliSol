import gymnasium as gym
import numpy as np
from solitaire_env import SpiderSolitaireEnv
from dqn_agent import DQNAgent

def preprocess_obs(obs_dict):
    columns = np.array(obs_dict["columns"], dtype=np.float32).flatten()
    lengths = np.array(obs_dict["column_lengths"], dtype=np.float32)
    deck_size = np.array([obs_dict["deck_size"]], dtype=np.float32)

    return np.concatenate([columns, lengths, deck_size])

env = SpiderSolitaireEnv()
obs, _ = env.reset()
obs_vec = preprocess_obs(obs)
obs_size = obs_vec.shape[0]
action_size = env.action_space.n

agent = DQNAgent(obs_size=obs_size, action_size=action_size)

episodes = 500
MAX_STEPS = 500

for ep in range(episodes):
    obs, _ = env.reset()
    obs = preprocess_obs(obs)
    total_reward = 0

    done = False
    for step in range(MAX_STEPS):
        action = agent.act(obs)
        next_obs, reward, done, _, _ = env.step(action)
        next_obs = preprocess_obs(next_obs)

        agent.remember(obs, action, reward, next_obs, done)
        agent.replay()

        obs = next_obs
        total_reward += reward
        
        if done:
            break

    print(f"Ep {ep+1}: Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
