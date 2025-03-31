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

Run the training script with the desired algorithm:

For Q-learning:
```sh
python train.py --algorithm q-learning
```

For SARSA:
```sh
python train.py --algorithm sarsa
```

### Visualizing the Results
After training, you can visualize the agent’s learned policy using:

```sh
python visualize.py
```

## Customization
You can modify the grid size, reward structure, and hyperparameters in the `config.py` file to experiment with different settings.

---

Feel free to contribute by opening issues or submitting pull requests!

