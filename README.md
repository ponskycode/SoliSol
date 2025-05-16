# SoliSol, an AI Solitaire Solver

The aim of the project is to create an AI agent that learns through reinforcement to solve Spider Solitaire.
  
The Demo version has the following changes:
- Number of columns: 6
- Drawing cards possible
- One color of cards
- Card ranks: 6
- Card sets: 4 (24 total cards)
- Movement rules: moving single cards or descending sequences to other columns
- Complete sequence (6->1) is removed from the game
- All cards are visible
- Legal actions are known to the agent from the start


## Project files

### `solitaire_env.py`
Contains the definition of the Spider Solitaire game environment in a simplified version, compatible with the Gymnasium interface. The file implements:
- game logic (dealing cards, moves, checking victory),
- state and action space,
- reward function,
- function rendering the current state of the game in the console.

### `dqn_agent.py`
Contains the implementation of the DQN (Deep Q-Network) agent in PyTorch. The agent:
- learns based on experiences saved in the buffer (replay buffer),
- uses a neural network to predict the value of Q,
- applies the epsilon-greedy policy for exploration and exploitation,
- updates its parameters based on mini-batches of experiences.

### `train_dqn.py`
Script running the DQN agent training process in the Spider Solitaire environment. Responsible for:
- initializing the environment and agent,
- processing observations to vector format,
- main training loop (episodes, steps, agent updates),
- printing training progress (reward, epsilon value).


## First simulations

### `Data:`

- Ep. 1-75: reward ~ -100
- Ep. 75-175: improvement to ~ -80/-70
- Ep. 175-500: stabilization around -75/-65

### `Probable causes of stagnation`

- Too general view of the game state
- Weak reward signal
- Too many possible actions / imprecize movement selection

### `Potentail solutions`

Short term:

- Add a bigger penalty for empty moves (e.g. -0.5 for an action that doesn't change anything).

- Add an explicit reward for completing a sequence (e.g. +10).

- Add a step limit if you don't already have one (e.g. MAX_STEPS = 200).

Medium term:

- Change the observation to e.g. one-hot for cards (6 ranks x 4 columns -> 24-bit for one column).

- Consider other action representations — instead of the action index, use (from, to, N), or build a legal_moves -> index mapping.