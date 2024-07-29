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
num_actions = 5  # jump, up, down, left, right
Q_table = np.zeros((num_states, num_actions))

# Action mapping
actions = ["jump", "up", "down", "left", "right"]

# Function to convert state binary vector to integer index
def state_to_index(state):
    platform = int(state[:5], 2)
    direction = int(state[5:], 2)
    return platform * 4 + direction

# Initialize socket connection
s = cn.connect(2037)

# Training loop
for episode in range(num_episodes):
    done = False
    state, _ = s.get_state_reward(s, "reset")  # Initialize state, assuming "reset" action resets the game
    state_idx = state_to_index(state)

    while not done:
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, num_actions - 1)  # Explore
        else:
            action_idx = np.argmax(Q_table[state_idx])  # Exploit

        action = actions[action_idx]
        next_state, reward = cn.get_state_reward(cn, action)
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


 