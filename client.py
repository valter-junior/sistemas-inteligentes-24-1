import connection as cn
import numpy as np
import random


# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table
num_states = 24 * 4  # 24 platforms * 4 directions
num_actions = 3  # jump, rotate left, rotate right
Q_table = np.zeros((num_states, num_actions))

# Action mapping
actions = ["jump", "rotate left", "rotate right"]

# Function to convert state binary vector to integer index
def state_to_index(state):
    platform = int(state[:5], 2)
    direction = int(state[5:], 2)
    return platform * 4 + direction

# Function to handle the state transition if the agent falls off
def handle_fall_off(state, reward):
    if reward == -1:
        platform = 0  # Reset to platform 0
        direction = state[5:]  # Keep the same direction
        state = f"{platform:05b}" + direction
    return state

# Initialize socket connection
s = cn.connect(2037)

# Training loop
for episode in range(num_episodes):
    done = False
    
    # Initialize state: platform 0, direction North (00)
    state = f"{0:05b}" + "00"
    state_idx = state_to_index(state)

    while not done:
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, num_actions - 1)  # Explore
        else:
            action_idx = np.argmax(Q_table[state_idx])  # Exploit

        action = actions[action_idx]
        next_state, reward = cn.get_state_reward(cn, action)
        next_state = handle_fall_off(next_state, reward)  # Handle fall off
        next_state_idx = state_to_index(next_state)

        # Q-learning update
        best_next_action = np.argmax(Q_table[next_state_idx])
        Q_table[state_idx, action_idx] = Q_table[state_idx, action_idx] + alpha * (
            reward + gamma * Q_table[next_state_idx, best_next_action] - Q_table[state_idx, action_idx]
        )

        state_idx = next_state_idx
        if next_state == "terminal_state":  # Define your terminal state condition
            done = True

    # Decay epsilon over time
    epsilon = max(0.01, epsilon * 0.995)

# Save the Q-table
np.save("q_table.npy", Q_table)
 