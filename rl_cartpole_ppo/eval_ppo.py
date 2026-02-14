import os
import numpy as np
import gymnasium as gym
import torch
from torch.distributions import Categorical

from ppo_agent import PPOAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppo_cartpole.pt")

@torch.no_grad()
def greedy_action(agent, state_np):
    s = torch.tensor(state_np, dtype=torch.float32, device=agent.device).unsqueeze(0)
    logits = agent.actor(s)
    return int(torch.argmax(logits, dim=1).item())

def evaluate(render=False, episodes=10):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    agent.load(MODEL_PATH)

    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=2000 + ep)
        total = 0
        for _ in range(500):
            action = greedy_action(agent, state)  # deterministic eval
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(int(total))

    env.close()
    print("Eval scores:", scores)
    print("Eval avg:", sum(scores) / len(scores))

if __name__ == "__main__":
    evaluate(render=False, episodes=10)
