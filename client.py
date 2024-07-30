import connection as cn
import numpy as np
import random as rd

# Connect to the game server
s = cn.connect(2037)

# Load the Q-table
q_table = np.loadtxt(r'C:\Users\Arthur\github\SI\sistemas-inteligentes-24-1\resultado.txt')
np.set_printoptions(precision=6)

# Define actions
actions = ["left", "right", "jump"]

# Parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.8  # Initial exploration rate

# Initialize state and reward
curr_state = 0
curr_reward = -14

def choose_action(epsilon, actions, curr_state):
    if rd.random() < epsilon:
        action = actions[rd.randint(0, 2)]
        print(f'Ação aleatória escolhida para o estado {curr_state}: {action}')
    else:
        number = np.argmax(q_table[curr_state])
        action = actions[number]
        print(f'Melhor ação escolhida para o estado {curr_state}: {action}')
    return action

def bellman_equation(r, s_prime, gamma):
    max_q_prime = np.max(q_table[s_prime])
    pontos = r + gamma * max_q_prime
    return pontos

# Training loop
while True:
    action = choose_action(epsilon, actions, curr_state)
    
    # Decay epsilon
    if epsilon > 0.1:
        epsilon -= 0.001
    print(f'epsilon: {epsilon}')
    
    # Map action to column
    if action == "left":
        col_action = 0
    elif action == "right":
        col_action = 1
    else:
        col_action = 2

    # Get next state and reward from the server
    state, reward = cn.get_state_reward(s, action)
    print(f'Action: {action}, Reward: {reward}, Next State: {state}')
    
    # Process state
    state = state[2:]
    next_state = int(state, 2)
    
    # Update Q-table
    print(f'Valor anterior dessa ação: {q_table[curr_state][col_action]}')
    q_table[curr_state][col_action] += alpha * (bellman_equation(reward, next_state, gamma) - q_table[curr_state][col_action])
    
    # Update current state and reward
    curr_state = next_state
    curr_reward = reward
    
    # Save the Q-table
    np.savetxt(r'C:\Users\Arthur\github\SI\sistemas-inteligentes-24-1\resultado.txt', q_table, fmt="%f")
