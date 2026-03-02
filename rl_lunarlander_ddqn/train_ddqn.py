import os
import numpy as np
import gymnasium as gym

from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ddqn_lunarlander.pt")

def greedy_eval(agent, episodes=5):
    env = gym.make("LunarLander-v3")
    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=1000 + ep)
        total = 0.0
        for _ in range(1000):
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return float(np.mean(scores))

def train():
    env = gym.make("LunarLander-v3")

    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n             # 4

    agent = DQNAgent(state_dim, action_dim, lr=5e-4, gamma=0.99)
    buffer = ReplayBuffer(capacity=200_000)

    episodes = 3000
    batch_size = 128
    min_buffer = 10_000
    target_update_every = 1000  # steps

    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.05

    total_steps = 0
    best_eval = -1e9

    for ep in range(episodes):
        state, info = env.reset()
        ep_reward = 0.0

        for _ in range(1000):
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

        if ep % 50 == 0:
            eval_avg = greedy_eval(agent, episodes=5) if len(buffer) >= min_buffer else None
            if eval_avg is not None:
                if eval_avg > best_eval:
                    best_eval = eval_avg
                    agent.save(MODEL_PATH)
                    print(f"Ep {ep} reward={ep_reward:.1f} eps={epsilon:.2f} eval={eval_avg:.1f} BEST saved")
                else:
                    print(f"Ep {ep} reward={ep_reward:.1f} eps={epsilon:.2f} eval={eval_avg:.1f} best={best_eval:.1f}")
            else:
                print(f"Ep {ep} reward={ep_reward:.1f} eps={epsilon:.2f} buffer={len(buffer)}")

        # Stop early if solved
        if best_eval >= 200.0:
            print("Solved (eval >= 200). Stopping early.")
            break

    env.close()
    print("Training finished. Best eval:", best_eval)
    print("Model path:", MODEL_PATH)

if __name__ == "__main__":
    train()
