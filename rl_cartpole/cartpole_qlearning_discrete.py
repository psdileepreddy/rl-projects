import numpy as np
import gymnasium as gym

def discretize(state, bins):
    """Convert continuous state -> discrete index tuple."""
    cart_pos, cart_vel, pole_angle, pole_ang_vel = state

    # Clip to reasonable ranges (CartPole can explode)
    cart_pos = np.clip(cart_pos, -2.4, 2.4)
    cart_vel = np.clip(cart_vel, -3.0, 3.0)
    pole_angle = np.clip(pole_angle, -0.2095, 0.2095)  # about 12 degrees
    pole_ang_vel = np.clip(pole_ang_vel, -3.5, 3.5)

    idxs = []
    for val, b in zip([cart_pos, cart_vel, pole_angle, pole_ang_vel], bins):
        idxs.append(int(np.digitize(val, b)))
    return tuple(idxs)

def epsilon_greedy(Q, s, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return int(np.argmax(Q[s]))

def main():
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n  # 2 actions

    # Define bins for each state dimension
    bins = [
        np.linspace(-2.4, 2.4, 8)[1:-1],      # cart position
        np.linspace(-3.0, 3.0, 8)[1:-1],      # cart velocity
        np.linspace(-0.2095, 0.2095, 10)[1:-1],  # pole angle
        np.linspace(-3.5, 3.5, 10)[1:-1],     # pole angular velocity
    ]

    # Q-table shape: (bins+1 for each dim) x actions
    q_shape = tuple(len(b) + 1 for b in bins) + (n_actions,)
    Q = np.zeros(q_shape, dtype=np.float32)

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05

    episodes = 2000
    max_steps = 500

    best_score = 0

    for ep in range(episodes):
        state, info = env.reset()
        s = discretize(state, bins)
        total_reward = 0

        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, n_actions)
            next_state, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            s_next = discretize(next_state, bins)

            # Q-learning update
            best_next = 0.0 if done else np.max(Q[s_next])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[s + (a,)]
            Q[s + (a,)] += alpha * td_error

            s = s_next
            total_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        best_score = max(best_score, int(total_reward))

        if ep % 100 == 0:
            print(f"Episode {ep}, score={int(total_reward)}, best={best_score}, epsilon={epsilon:.2f}")

    env.close()
    print("Training done. Best score:", best_score)
    # ---- QUICK EVALUATION (TEMPORARY) ----
    env = gym.make("CartPole-v1")

    scores = []
    for _ in range(10):
        state, info = env.reset()
        s = discretize(state, bins)
        total = 0

        for _ in range(500):
            a = int(np.argmax(Q[s]))
            next_state, reward, terminated, truncated, info = env.step(a)
            s = discretize(next_state, bins)
            total += reward
            if terminated or truncated:
                break

        scores.append(int(total))

    env.close()
    print("Eval scores:", scores)
    print("Eval avg:", sum(scores) / len(scores))

if __name__ == "__main__":
    main()
