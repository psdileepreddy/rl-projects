import gymnasium as gym

env = gym.make("CartPole-v1")

state, info = env.reset()
print("Initial state:", state)

total_reward = 0

for step in range(200):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    print(f"Step {step}, Action {action}, Reward {reward}")

    if terminated or truncated:
        print("Episode ended")
        break

print("Total reward:", total_reward)
env.close()
