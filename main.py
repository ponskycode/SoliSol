from solitaire_env import *
from tests import *
from train_dqn import DQNTrainer

def main():
    env = SpiderSolitaireEnv()
    manual_test(env)
    trainer = DQNTrainer(env=env, episodes=50, max_steps=500)
    trainer.train()

if __name__ == "__main__":
    main()