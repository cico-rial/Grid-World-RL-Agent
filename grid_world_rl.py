import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import argparse
import tqdm
import math

def parsing_arguments():

    parser = argparse.ArgumentParser(
        description=f"Launching the grid-world with customized commands."
    )
      
    parser.add_argument('--train', '-t', type=int, default=1, help="Specify the number of episodes to train your agent with.\n"
    "Default: 1")
    parser.add_argument('--grid-size', '-n', type=int, default=get_q_table_metadata().get('N'), help="Specify the size of the N-grid world.\n"
    "Default: 3")
    parser.add_argument('--reset', '-r', type=lambda x: (str(x).lower() == "true"), default=False, help="Specify whether to reset the learning to an initial state.\n"
    "Default: False")
    parser.add_argument('--refresh-rate', '-rr', type=float, default=4, help="Specify the refresh rate (in seconds) of the terminal. It affects the time to wait between the iterations.\n"
    "Defualt: 4")
    parser.add_argument('--starting-point', '-sp', type=int, default=0, help="Specify the initial position of the player in the grid-world.\n"
    "Default: 0")
    parser.add_argument('--algorithm', '-a', default="q-learning", help="Specify the algorithm to use for make the agent learn.\n"
    "Choices: ['q-learning','sarsa']\n"
    "Default: 'q-learning'")
    
    # Disallow unrecognized arguments
    args = parser.parse_args()

    # here i can put all the logic i want!
    if args.starting_point >= (args.grid_size)**2 - 1:
        parser.error(f"-starting-point must be between 0 and {args.grid_size}, but got {args.starting_point}")

    return args

def print_maze(maze):
    maze_representation = ""
    count = 0
    for x in maze:
        if count != N:
            maze_representation += str(x) + "  "
            count += 1
        else:
            maze_representation += f"\n{str(x)}  "
            count = 1
    print("")
    print(maze_representation)
    print("")

def reset_position(maze, visited_states, learning_history):
    maze = [blank_position for i in range(0,N**2)]
    maze[initial_state] = player
    maze[final_state] = goal
    visited_states = [initial_state]
    learning_history = [maze.copy()]
    return maze, visited_states, learning_history

def get_q_table_metadata():
    metadata = {"N": 3, "episodes_trained": 0} # default value in case the json file has never created before.
    if os.path.exists(q_table_name):
        with open(q_table_name, "r") as file:
            try:
                json_data = json.load(file)
                metadata = json_data.get('metadata')
            except json.JSONDecodeError:
                print("Error in decoding the jsonfile. Setting the default value for N (3)")
    return metadata

def get_hyperparameters(file_name):
    config = {
    "epsilon" : 0.05,
    "goal_reward" : 1,
    "wall_reward" : -1,
    "repeated_state_reward" : -1,
    "step_reward" : -0.1,
    "gamma" : 0.9,
    "alpha" : 0.5
}
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            try:
                config = json.load(file)
            except json.JSONDecodeError:
                print("Error in decoding the config. Setting the default values for the hyperparameters.")
                print(config)
    else:
        with open(file_name, "w") as file:
            json.dump(config, file, indent=4)
        print("Failed to import the saved config.json")
        print("The file has never been created before.")
        print("Not a problem, I'm building a new one!")
        time.sleep(refresh_rate)
        _ = os.system("cls")
    
    return config



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


def build_q_table(N, reset: bool):
    
    positions = range(0,N**2)
    q_table = {position: {action: 0 for action in actions} for position in positions}
    q_table['metadata'] = {"N": N,
                           "episodes_trained": 0}

    if os.path.exists(q_table_name) and not reset:
        with open(q_table_name, "r") as file:
            try:
                json_data = json.load(file)
                assert len(json_data.keys()) - 1 == N**2 # checking whether the current size of the board matches the previous saved one.
                # -1 because i also store the metadata.
                q_table = {int(k): v for k,v in json_data.items() if k != "metadata"}
                q_table['metadata'] = json_data.get('metadata')
            except json.JSONDecodeError:
                print("Error in decoding the jsonfile.")  
            except AssertionError:
                print("Failed to import the saved grid-world.")
                print("The grid size has changed.")
                print("Not a problem, I'm building a new one!")
                time.sleep(refresh_rate)
                _ = os.system("cls")
                # print(f"old:{len(json_data.keys())} new:{N**2} ")
    else:
        if reset:
            print("Requested reset for the grid-world.")
            print("Not a problem, I'm building a new one!")
            time.sleep(refresh_rate)
            _ = os.system("cls")
        else:
            print("Failed to import the saved grid-world.")
            print("The file has never been created before.")
            print("Not a problem, I'm building a new one!")
            time.sleep(refresh_rate)
            _ = os.system("cls")

    return q_table

def save_q_table(q_table):
    try:
        with open(q_table_name, "w") as file:
            json.dump(q_table, file, indent=4)
        print("")
        print("Q-table succesfully saved.")
    except:
        print("")
        print("Failed to save the Q-table.")
        print("Not much to say about it I'm sorry :(")


def choose_action(current_state, verbose=True):
    # let's choose an action with ε greedy
    if verbose:
        print(f"q scores: {q_table.get(current_state)}")
        
    # epsilon = 0.05
    epsilon = get_hyperparameters('config.json').get('epsilon')
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
    maze[current_state] = stepped_position
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

    _ = os.system("cls") #??

    q_table_name = "q_table.json"

    args = parsing_arguments()

    episodes = args.train
    N = args.grid_size ## bro you can set the default value as the length of last-saved q_table
    reset = args.reset
    refresh_rate = args.refresh_rate
    initial_state = args.starting_point
    final_state = N**2 - 1
    training = (episodes != 1)
    algorithm = args.algorithm

    player = "🏃"
    goal = "🚩" 
    blank_position = "🌱" 
    stepped_position = '👣' 
    
    maze = [blank_position for i in range(0,N**2)]
    maze[initial_state] = player
    maze[final_state] = goal
    visited_states = [initial_state]

    actions = ["up", "down", "left", "right"]

    learning_history = [maze.copy()]

    config = get_hyperparameters('config.json')

    goal_reward = config.get('goal_reward')
    wall_reward = config.get('wall_reward')
    repeated_state_reward = config.get('repeated_state_reward')
    step_reward = config.get('step_reward')
    gamma = config.get('gamma')
    alpha = config.get('alpha')


    transition_model = build_transition_model(N)
    q_table = build_q_table(N, reset=reset) 

    if not training:
        print(f"Grid size: {N}x{N}")
        print(f"Algorithm: {algorithm}")
        print(f"Number of episodes trained (up to now): {q_table.get('metadata').get('episodes_trained')}")
    else:
        print(f"Grid size: {N}x{N}")
        print(f"Training the agent for {episodes} episodes to find the exit in the grid-world.")
        print(f"Learning the q-table with {algorithm} algorithm.")
        print(f"")
        time.sleep(refresh_rate)


    start_time = time.time()
    total_iterations = 0
    for episode in tqdm.tqdm(range(episodes), disable=not training):

        maze, visited_states, learning_history = reset_position(maze, visited_states, learning_history)

        training_time = range(0,1000)
        current_state = initial_state
        if not training:
            print_maze(maze)
            print("timestep: 0")
            print(f"I'm in position {current_state} and the episode just started.")

        current_action, old_q_estimate = choose_action(current_state, verbose=(not training))

        if not training:
            time.sleep(refresh_rate)
        for t in training_time: # for each episode 
            
            new_state = transition_model.get(current_state).get(current_action)  
            reward, reward_message = get_reward(current_state, new_state)
            move_player(current_state, new_state)
            
            if not training:
                _ = os.system("cls")
                print(f"Grid size: {N}x{N}")
                print(f"Algorithm: {algorithm}")
                print(f"Number of episodes trained (up to now): {q_table.get('metadata').get('episodes_trained')}")
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

            if current_state == final_state:
                break

            if not training:
                time.sleep(refresh_rate)
            
        total_iterations += t+1
        q_table['metadata']['episodes_trained'] += 1        

        if not training:
            print(f"Episode time (iterations): {t+1}")

    end_time = time.time()
    total_time = round(end_time - start_time, 3)
    save_q_table(q_table)

    if training:
        print
        print(f"Succesfully trained! Now your agent should be able to find the exit more easily!")
        print(f"Training time: {total_time} s ({total_iterations} iterations)")