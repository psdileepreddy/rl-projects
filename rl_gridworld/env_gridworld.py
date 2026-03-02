import numpy as np

class GridWorld:
    def __init__(self):
        self.rows = 3
        self.cols = 3

        self.start_state = (0, 0)
        self.goal_state = (2, 2)
        self.obstacle = (1, 1)

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

        # out of bounds
        if (next_row < 0 or next_row >= self.rows or
            next_col < 0 or next_col >= self.cols):
            next_state = self.state
            reward = -1

        # obstacle
        elif next_state == self.obstacle:
            next_state = self.state
            reward = -5

        # goal
        elif next_state == self.goal_state:
            reward = 1
            done = True


        self.state = next_state
        return next_state, reward, done

    def state_to_index(self, state):
        r, c = state
        return r * self.cols + c

    def num_states(self):
        return self.rows * self.cols

    def num_actions(self):
        return len(self.actions)
