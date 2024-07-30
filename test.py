import connection as cn
import numpy as np
import random
import socket

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1  # Exploration rate for initial exploration

# Initialize Q-table
num_states = 24 * 4  # 24 platforms * 4 directions
num_actions = 3  # left, right, jump
Q_table = np.zeros((num_states, num_actions))

# Action mapping
actions = ["left", "right", "jump"]

# Function to convert state binary vector to integer index
def state_to_index(state):
    platform = int(state[:5], 2)
    direction = int(state[5:], 2)
    return platform * 4 + direction

# Function to convert index back to state binary vector
def index_to_state(index):
    platform = index // 4
    direction = index % 4
    return f"{platform:05b}{direction:02b}"

# Function to check if the goal state is reached
def check_goal_state(state):
    platform = int(state[:5], 2)
    if platform in [13, 23]:
        return True
    return False

# Pre-fill Q-table for platform 0
initial_state_index = state_to_index("0000000")
for direction in range(1, 4):  # Directions: East, South, West
    Q_table[initial_state_index + direction, 2] = -100  # Jump leads to fall

# Initialize socket connection
s = cn.connect(2037)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    done = False
    
    # Initialize state: platform 0, direction North (00)
    state = f"{0:05b}" + "00"
    state_idx = state_to_index(state)

    while not done:
        if Q_table[state_idx, 2] == 0:  # If 'jump' Q-value is zero, explore it
            action_idx = 2  # Choose 'jump'
        else:
            action_idx = np.argmax(Q_table[state_idx])  # Exploit: Select action with highest Q-value

        action = actions[action_idx]
        
        # Get next state and reward from the server
        next_state, reward = cn.get_state_reward(s, action)
        
        # Check if agent has fallen off
        if reward == -100:
            next_state_idx = state_to_index("0000000")
            done = True  # End the episode
        
        # Check if goal state is reached
        elif check_goal_state(next_state):
            reward = 100  # High reward for reaching goal
            next_state_idx = state_to_index(next_state)
            done = True
        else:
            next_state_idx = state_to_index(next_state)

        # Q-learning update
        Q_table[state_idx, action_idx] = (1 - alpha) * Q_table[state_idx, action_idx] + alpha * (
            reward + gamma * np.max(Q_table[next_state_idx])
        )
        
        # Update state for next iteration
        state_idx = next_state_idx
        state = next_state

# Save the Q-table
np.save("q_table.npy", Q_table)
