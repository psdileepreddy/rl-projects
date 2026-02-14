import numpy as np

class GridWorld:
    def __init__(self):
        # Grid size
        self.rows = 3
        self.cols = 3

        # Start and goal positions
        self.start_state = (0, 0)
        self.goal_state = (2, 2)

        # Obstacle position
        self.obstacle = (1, 1)

        # Action space: up, down, left, right
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        row, col = self.state
        dr, dc = self.actions[action]

        next_row = row + dr
        next_col = col + dc
        next_state = (next_row, next_col)

        reward = -1
        done = False

        # Out of bounds
        if (next_row < 0 or next_row >= self.rows or
            next_col < 0 or next_col >= self.cols):
            next_state = self.state
            reward = -1

        # Obstacle
        elif next_state == self.obstacle:
            next_state = self.state
            reward = -5

        # Goal
        elif next_state == self.goal_state:
            reward = 10
            done = True

        self.state = next_state
        return next_state, reward, done

    # --- Helpers for Q-table ---
    def state_to_index(self, state):
        r, c = state
        return r * self.cols + c

    def num_states(self):
        return self.rows * self.cols

    def num_actions(self):
        return len(self.actions)

def epsilon_greedy(Q, state_idx, epsilon):
    if np.random.rand() < epsilon:
        # Explore: random action
        return np.random.choice(Q.shape[1])
    else:
        # Exploit: best known action
        return np.argmax(Q[state_idx])
def print_policy(env, Q):
    arrows = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→"
    }

    print("\nLearned Policy (best action in each state):")
    for r in range(env.rows):
        row_symbols = []
        for c in range(env.cols):
            state = (r, c)

            if state == env.obstacle:
                row_symbols.append("X")
                continue
            if state == env.goal_state:
                row_symbols.append("G")
                continue
            if state == env.start_state:
                # show action too, but keep S label
                s_idx = env.state_to_index(state)
                best_a = int(np.argmax(Q[s_idx]))
                row_symbols.append("S" + arrows[best_a])
                continue

            s_idx = env.state_to_index(state)
            best_a = int(np.argmax(Q[s_idx]))
            row_symbols.append(arrows[best_a])

        print("  ".join(row_symbols))

if __name__ == "__main__":
    env = GridWorld()

    Q = np.zeros((env.num_states(), env.num_actions()))

    # Hyperparameters
    alpha = 0.1      # learning rate
    gamma = 0.9      # discount factor
    epsilon = 1.0    # start with full exploration
    epsilon_decay = 0.995
    min_epsilon = 0.1
    episodes = 500


    for episode in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        done = False
        total_reward = 0


        while not done:
            action = epsilon_greedy(Q, state_idx, epsilon)

            next_state, reward, done = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            # Q-learning update (TD learning)
            best_next_q = np.max(Q[next_state_idx])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state_idx, action]
            Q[state_idx, action] += alpha * td_error

            state_idx = next_state_idx
            total_reward += reward


        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Progress print
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
def print_greedy_path(env, Q, max_steps=20):
    state = env.reset()
    path = [state]

    print("\nGreedy path from Start to Goal:")

    for step in range(max_steps):
        s_idx = env.state_to_index(state)
        action = int(np.argmax(Q[s_idx]))

        next_state, reward, done = env.step(action)
        path.append(next_state)

        print(f"Step {step}: State {state} -> Action {action} -> Next {next_state}")

        state = next_state

        if done:
            print("Reached the goal!")
            break

    print("\nPath taken:")
    print(path)
print_policy(env, Q)
print_greedy_path(env, Q)
