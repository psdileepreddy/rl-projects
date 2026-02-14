import numpy as np
import gymnasium as gym

# --- SAME discretize function as training ---
def discretize(state, bins):
    cart_pos, cart_vel, pole_angle, pole_ang_vel = state

    cart_pos = np.clip(cart_pos, -2.4, 2.4)
    cart_vel = np.clip(cart_vel, -3.0, 3.0)
    pole_angle = np.clip(pole_angle, -0.2095, 0.2095)
    pole_ang_vel = np.clip(pole_ang_vel, -3.5, 3.5)

    idxs = []
    for val, b in zip([cart_pos, cart_vel, pole_angle, pole_ang_vel], bins):
        idxs.append(int(np.digitize(val, b)))
    return tuple(idxs)


def evaluate(Q, bins, episodes=10, max_steps=500):
    env = gym.make("CartPole-v1")
    scores = []

    for ep in range(episodes):
        state, info = env.reset()
        s = discretize(state, bins)
        total_reward = 0

        for _ in range(max_steps):
            action = int(np.argmax(Q[s]))  # greedy (no exploration)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            s = discretize(next_state, bins)

            if terminated or truncated:
                break

        scores.append(int(total_reward))

    env.close()
    print("Evaluation scores:", scores)
    print("Average score:", sum(scores) / len(scores))


if __name__ == "__main__":
    print("This file evaluates a trained Q-table.")
    print("Paste your Q-table and bins here before running.")
