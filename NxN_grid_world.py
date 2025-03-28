import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import argparse

def parsing_arguments():

    parser = argparse.ArgumentParser(
        description=f"Launching the program with customized commands."
    )
      
    parser.add_argument('--train', '-t', type=int, nargs="?", help="whether to train the agent or not.")

    # Disallow unrecognized arguments
    args = parser.parse_args()

    return args

def print_maze(maze):
    print("")
    print(maze[:N])
    for i in range(1,N-1):
        print(maze[i*N:(i+1)*N])
    print(maze[N**2-N:])
    print("")


def reset_position(maze, visited_states, learning_history):
    maze = [0 for i in range(0,N**2)]
    maze[initial_state] = player
    maze[final_state] = goal
    visited_states = [initial_state]
    learning_history = [maze.copy()]
    return maze, visited_states, learning_history

def build_transition_model(N):
    positions = range(0,N**2)
    transition_model = {position: {} for position in positions}

    for position in positions:
        transition_model[position]["up"] = position - N
        transition_model[position]["down"] = position + N
        transition_model[position]["right"] = position + 1
        transition_model[position]["left"] = position - 1
        if position < N:
            transition_model[position]["up"] = position
        if position in range(N**2 - N, N**2):
            transition_model[position]["down"] = position
        if position % N == 0:
            transition_model[position]["left"] = position
        if (position + 1) % N == 0:
            transition_model[position]["right"] = position
    
    return transition_model


def build_q_table(N):
    positions = range(0,N**2)
    q_table = {position: {action: 0 for action in actions} for position in positions}
    return q_table


def choose_action(current_state, verbose=True):
    # let's choose an action with ε greedy
    if verbose:
        print(f"q scores: {q_table.get(current_state)}")
        
    epsilon = 0.05
    sample = random.uniform(0,1)
 
    next_action = random.choice(actions) # setting an initial random action
    best_q_value = q_table.get(current_state).get(next_action) # picking the q value associated to the action 

    for action in actions:
        if q_table.get(current_state).get(action) > best_q_value:
            next_action = action
            best_q_value = q_table.get(current_state).get(next_action)
    
    # for now my next_action is the best according to greedy selection.
    
    if sample > epsilon:
        # greedy case
        next_action = next_action
        if verbose:
            print(f"selected action: {next_action} (greedy)")
            
        
    else:
        actions_without_the_best = [action for action in actions if action != next_action]
        next_action = random.choice(actions_without_the_best)
        best_q_value = q_table.get(current_state).get(next_action)
        if verbose:
            print(f"selected action: {next_action} (random)")
    
    return next_action, best_q_value
    

def get_reward(current_state, new_state):
    if new_state == final_state:
        reward = goal_reward
        reward_message = "I reached the goal!"

    elif current_state == new_state:
        reward = wall_reward
        reward_message = "I hit the wall!"

    elif new_state in visited_states:
        reward = repeated_state_reward
        reward_message = "I have already visited this state!"
    
    elif current_state != new_state:
        reward = step_reward
        reward_message = "I made a step!"

    else:
        print("what is going on???")
        exit(0)
        
    
    return reward, reward_message

def move_player(current_state, new_state):
    maze[current_state] = 0
    maze[new_state] = player
    learning_history.append(maze.copy())
    if new_state not in visited_states:
        visited_states.append(new_state)


def get_best_q_value(current_state):
    next_action = actions[0]
    best_q_value = q_table.get(current_state).get(next_action) # picking the q value associated to the action 

    for action in actions[1:]:
        if q_table.get(current_state).get(action) > best_q_value:
            next_action = action
            best_q_value = q_table.get(current_state).get(next_action)
    return best_q_value


if __name__ == "__main__":

    _ = os.system("cls")
    args = parsing_arguments()

    episodes = args.train
    try:
        episodes = int(episodes)
        training=True
    except:
        print(f"No training. {episodes}")
        episodes = 1
        training=False
       

    N = 3

    player = "P"
    goal = "G"
    initial_state = 0
    final_state = N**2 - 1


    maze = [0 for i in range(0,N**2)]
    maze[initial_state] = player
    maze[final_state] = goal
    visited_states = [initial_state]

    actions = ["up", "down", "left", "right"]

    learning_history = [maze.copy()]
        
    goal_reward = 1
    wall_reward = -1
    repeated_state_reward = -1
    step_reward = -0.1

    gamma = 0.9
    alpha = 0.5


    transition_model = build_transition_model(N)

    q_table_name = "q_table.json"

    if os.path.exists(q_table_name):
        with open(q_table_name, "r") as file:
            try:
                json_data = json.load(file)
                q_table = {int(k): v for k,v in json_data.items()}
            except json.JSONDecodeError:
                print("Error in decoding the jsonfile.")
                q_table = build_q_table(N)  
    else:
        q_table = build_q_table(N)


    # algorithm = "sarsa"
    algorithm = "q-learning"

    print(f"Running the {algorithm} algorithm")

    for episode in range(episodes):

        maze, visited_states, learning_history = reset_position(maze, visited_states, learning_history)

        print_maze(maze)
        print("timestep: 0")

        training_time = range(0,1000)
        current_state = initial_state
        print(f"I'm in position {current_state} and the episode just started.")

        current_action, old_q_estimate = choose_action(current_state)
        # print("")

        if not training:
            time.sleep(4)
        for t in training_time: # for each episode 
            
            new_state = transition_model.get(current_state).get(current_action)  
            reward, reward_message = get_reward(current_state, new_state)
            move_player(current_state, new_state)
            
            _ = os.system("cls")
            print_maze(maze)
            
            print(f"timestep: {t+1}")
            print(f"I moved to the position {new_state} ({current_state}-->{new_state})")
            print(f"{reward_message} reward: {reward}")

            new_action, next_q_estimate = choose_action(new_state, verbose=((not new_state==final_state) and not training))

            if algorithm == "sarsa":
                updated_q_estimate = old_q_estimate + alpha*(reward + gamma*next_q_estimate - old_q_estimate)
            if algorithm == "q-learning":
                updated_q_estimate = old_q_estimate + alpha*(reward + gamma*get_best_q_value(current_state) - old_q_estimate)

            q_table[current_state][current_action] = updated_q_estimate

            current_state = new_state
            current_action, old_q_estimate = new_action, next_q_estimate

            if current_state == N**2-1:
                break

            if not training:
                time.sleep(4)

        with open(q_table_name, "w") as file:
            json.dump(q_table, file, indent=4)

        print("Q-table saved.")

        print("")
        print(f"Episode time (iterations): {t+1}")