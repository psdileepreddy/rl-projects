import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dqn_cartpole.pt")

import numpy as np
import gymnasium as gym

from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent

def greedy_eval(env, agent, episodes=5, max_steps=500):
    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=1000 + ep)
        total = 0
        for _ in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)  # greedy
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(int(total))
    return sum(scores) / len(scores)

def train(best_eval = 0):
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2

    agent = DQNAgent(state_dim, action_dim, lr=5e-4, gamma=0.99)
    buffer = ReplayBuffer(capacity=50_000)

    episodes = 2000
    batch_size = 128
    min_buffer = 1000
    target_update_every = 200  # steps
    max_steps = 500

    epsilon = 1.0
    epsilon_decay = 0.997
    min_epsilon = 0.02

    total_steps = 0
    best_eval = 0

    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= min_buffer:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            if total_steps % target_update_every == 0:
                agent.update_target()

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if ep % 20 == 0 and len(buffer) >= min_buffer:
            eval_avg = greedy_eval(env, agent, episodes=5)
            print(f"Greedy eval avg: {eval_avg:.1f} (best: {best_eval:.1f})")

            if eval_avg > best_eval:
                best_eval = eval_avg
                agent.save(MODEL_PATH)
                print("Saved NEW best model to", MODEL_PATH)

        if ep % 20 == 0:
            print(f"Episode {ep}, reward={int(ep_reward)}, epsilon={epsilon:.2f}, buffer={len(buffer)}")

    env.close()

    print("Training finished. Saved model to dqn_cartpole.pt")


if __name__ == "__main__":
    train()
