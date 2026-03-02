import os
import numpy as np
import gymnasium as gym

from dqn_agent import DQNAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ddqn_lunarlander.pt")

def evaluate(render=False, episodes=10):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.load(MODEL_PATH)

    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=2000 + ep)
        total = 0.0
        for _ in range(1000):
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)

    env.close()
    print("Eval scores:", [round(s, 1) for s in scores])
    print("Eval avg:", float(np.mean(scores)))

if __name__ == "__main__":
    evaluate(render=True, episodes=3)
