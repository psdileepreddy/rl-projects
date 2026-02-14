from env_gridworld import GridWorld
from agent_qlearning import QLearningAgent
from visualize import print_policy, print_greedy_path

def train():
    env = GridWorld()
    agent = QLearningAgent(env.num_states(), env.num_actions(), alpha=0.1, gamma=0.9)

    episodes = 3000
    max_steps = 50

    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.05

    for ep in range(episodes):
        state = env.reset()
        s_idx = env.state_to_index(state)
        total_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(s_idx, epsilon)
            next_state, reward, done = env.step(action)
            s_next_idx = env.state_to_index(next_state)

            agent.update(s_idx, action, reward, s_next_idx, done)

            s_idx = s_next_idx
            total_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if ep % 200 == 0:
            print(f"Episode {ep}, total_reward={total_reward}, epsilon={epsilon:.2f}")

    print_policy(env, agent.Q)
    print_greedy_path(env, agent.Q)

if __name__ == "__main__":
    train()
