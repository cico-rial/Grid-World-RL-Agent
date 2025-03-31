# Grid World RL Agent

This project implements an AI agent that learns to navigate a grid world and reach a goal using reinforcement learning. The agent can be trained using either **Q-learning** (off-policy) or **SARSA** (on-policy).

## Algorithms

### Q-learning (Off-policy)
Q-learning is a value-based reinforcement learning algorithm that updates its action-value function using the Bellman equation. The agent learns the optimal policy by selecting actions that maximize future rewards, regardless of the policy it follows during exploration.

### SARSA (On-policy)
SARSA is another value-based reinforcement learning algorithm, but it updates its action-value function based on the actual actions taken under the current policy. This means that it learns a policy while following it, making it more sensitive to exploration strategies.

## Setup Instructions

### Requirements
Ensure you have Python installed, along with the following dependencies:

```sh
pip install numpy matplotlib
```

### Running the Code
Clone the repository and navigate into the project folder:

```sh
git clone <repo_url>
cd grid-world-rl
```

You can run the script with custom commands:
```
usage: NxN_grid_world.py [-h] [--train TRAIN] [--grid-size GRID_SIZE] [--reset RESET] [--refresh-rate REFRESH_RATE] [--starting-point STARTING_POINT]
                         [--algorithm ALGORITHM]

Launching the grid-world with customized commands.

options:
  -h, --help            show this help message and exit
  --train TRAIN, -t TRAIN
                        Specify the number of episodes to train your agent with. Default: 1
  --grid-size GRID_SIZE, -n GRID_SIZE
                        Specify the size of the N-grid world. Default: 3
  --reset RESET, -r RESET
                        Specify whether to reset the learning to an initial state. Default: False
  --refresh-rate REFRESH_RATE, -rr REFRESH_RATE
                        Specify the refresh rate (in seconds) of the terminal. It affects the time to wait between the iterations. Defualt: 4
  --starting-point STARTING_POINT, -sp STARTING_POINT
                        Specify the initial position of the player in the grid-world. Default: 0
  --algorithm ALGORITHM, -a ALGORITHM
                        Specify the algorithm to use for make the agent learn. Choices: ['q-learning','sarsa'] Default: 'q-learning'
```

#### Training your agent
For training your agent, you can run the program with the following command:
```sh
python name.py --train 10
```

For testing your agent, you can simply run this basic command:
```sh
python name.py 
```

It's nice to note that you can launch the program by specifying custom options:
```sh
python name.py --grid-size 10 --refresh-rate 1 --starting-point 0 --algorithm sarsa
```
## Customization
You can modify the grid size, reward structure, and hyperparameters in the `config.py` file to experiment with different settings.

---

Feel free to contribute by opening issues or submitting pull requests!

