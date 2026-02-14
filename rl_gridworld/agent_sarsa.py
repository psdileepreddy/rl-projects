import numpy as np

class SARSAAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions

    def select_action(self, state_idx, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s_idx, action, reward, s_next_idx, next_action, done):
        next_q = 0.0 if done else self.Q[s_next_idx, next_action]
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.Q[s_idx, action]
        self.Q[s_idx, action] += self.alpha * td_error
