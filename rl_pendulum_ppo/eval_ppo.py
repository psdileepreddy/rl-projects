import os
import numpy as np
import gymnasium as gym
import torch

from ppo_agent import PPOAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppo_pendulum.pt")

@torch.no_grad()
def deterministic_action(agent, state):
    s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    mu, _ = agent.actor(s)
    a = mu.squeeze(0).cpu().numpy()
    return np.clip(a, agent.action_low, agent.action_high)

def evaluate(render=False, episodes=10):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]

    agent = PPOAgent(state_dim, action_dim, action_low, action_high)
    agent.load(MODEL_PATH)

    returns = []
    for ep in range(episodes):
        state, info = env.reset(seed=2000 + ep)
        total = 0.0
        for _ in range(200):
            action = deterministic_action(agent, state)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)

    env.close()
    print("Eval returns:", [round(r, 1) for r in returns])
    print("Eval avg:", float(np.mean(returns)))

if __name__ == "__main__":
    evaluate(render=False, episodes=10)
