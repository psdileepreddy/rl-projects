import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions

    def select_action(self, state_idx, epsilon):
        # epsilon-greedy
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s_idx, action, reward, s_next_idx, done):
        # Q-learning TD target
        best_next = 0.0 if done else np.max(self.Q[s_next_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[s_idx, action]
        self.Q[s_idx, action] += self.alpha * td_error
