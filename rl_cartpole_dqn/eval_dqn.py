import numpy as np
import gymnasium as gym
import numpy as np
np.random.seed(0)

from dqn_agent import DQNAgent


def evaluate(model_path="dqn_cartpole.pt", episodes=10, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    import os
    print("Current working directory:", os.getcwd())
    print("Looking for model at:", os.path.abspath(model_path))
    print("File exists:", os.path.exists(model_path))

    agent.load(model_path)

    scores = []
    for ep in range(episodes):

        state, info = env.reset(seed=ep)
        total = 0

        for _ in range(500):
            # pure greedy
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break

        scores.append(int(total))

    env.close()
    print("Eval scores:", scores)
    print("Eval avg:", sum(scores) / len(scores))


if __name__ == "__main__":
    evaluate(render=False)
