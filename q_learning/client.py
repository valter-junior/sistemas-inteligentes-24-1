import gymnasium as gym
import numpy as np


import env


class QLearningAgent:
    def __init__(
        self, observation_space, action_space, alpha=1e-1, gamma=0.99, strategy=0
    ):
        self.obs_size = observation_space.n
        self.action_size = action_space.n
        self.q_table = np.zeros((self.obs_size, self.action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.strategy = strategy
        self.epsilon = 0.15 if strategy == 0 else 0.8
        self.n_counter = np.zeros((self.obs_size, self.action_size))
        self.total_count = 0

    def get_output(self, observation):
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def get_action(self, observation):
        if self.strategy == 0:
            action = self.epsilon_greedy(observation)
        elif self.strategy == 1:
            action = self.epsilon_greedy_decay(observation)
        elif self.strategy == 2:
            action = self.softmax(observation)
        elif self.strategy == 3:
            action = self.ucb(observation)
        return action

    def epsilon_greedy(self, observation):
        if np.random.random() < 1 - self.epsilon:
            action = self.get_output(observation)
        else:
            action = np.random.randint(self.action_size)
        return action

    def epsilon_greedy_decay(self, observation):
        action = self.epsilon_greedy(observation)
        self.epsilon *= 0.997
        self.epsilon = max(0.05, self.epsilon)
        print(self.epsilon)
        return action

    def softmax(self, observation):
        temperature = 0.1
        logits = self.q_table[observation] / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        action = np.random.choice(self.action_size, p=probs)
        return action

    def ucb(self, observation):
        n = self.n_counter[observation].sum()
        c = 1
        if n == 0:
            action = np.random.randint(self.action_size)
            return action
        ucb = self.q_table[observation] + c * np.sqrt(np.log(self.total_count) / n)
        action = np.argmax(ucb)
        self.n_counter[observation][action] += 1
        return action

    def update_policy(self, observation, action, reward, next_obs):
        update = reward + self.gamma * (
            self.q_table[next_obs].max() - self.q_table[observation][action]
        )
        self.q_table[observation][action] = (
            self.q_table[observation][action] + self.alpha * update
        )
        self.total_count += 1


def main(strategy=0):
    env = gym.make("Amongois-v0")
    agent = QLearningAgent(env.observation_space, env.action_space, strategy=strategy)
    obs, _ = env.reset()
    reward = 0
    for _ in range(1000):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            if done:
                next_obs = obs
            agent.update_policy(obs, action, reward, next_obs)
            obs = next_obs
    # Wait until the plot window is closed
    env.close()


if __name__ == "__main__":
    arg = input(
        "Enter the strategy: \n 0: Epsilon Greedy \n 1: Epsilon Greedy Decay \n 2: Softmax \n 3: UCB \n"
    )
    strategy = int(arg)
    main(strategy=strategy)
