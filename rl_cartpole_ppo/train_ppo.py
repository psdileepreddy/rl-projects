import os
import numpy as np
import gymnasium as gym

from ppo_agent import PPOAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppo_cartpole.pt")

def collect_rollout(env, agent, rollout_steps=2048):
    states = []
    actions = []
    logps = []
    rewards = []
    values = []
    dones = []

    state, info = env.reset()
    for _ in range(rollout_steps):
        action, logp, value = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

        states.append(state)
        actions.append(action)
        logps.append(logp)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        state = next_state

        if terminated or truncated:
            state, info = env.reset()

    # bootstrap value from last state for GAE
    _, _, last_value = agent.act(state)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    logps = np.array(logps, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    adv, returns = agent.compute_gae(rewards, values, dones, last_value)

    return states, actions, logps, returns.astype(np.float32), adv.astype(np.float32)

def evaluate(env, agent, episodes=10):
    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=1000 + ep)
        total = 0
        for _ in range(500):
            action, _, _ = agent.act(state)  # stochastic; good enough for quick eval
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(int(total))
    return sum(scores) / len(scores)

def train():
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim, action_dim,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01
    )

    updates = 60
    rollout_steps = 2048
    epochs = 10
    minibatch_size = 256

    best_eval = 0.0

    for u in range(updates):
        batch = collect_rollout(env, agent, rollout_steps=rollout_steps)
        agent.update(batch, epochs=epochs, minibatch_size=minibatch_size)

        avg = evaluate(eval_env, agent, episodes=10)
        print(f"Update {u}, eval avg: {avg:.1f}")

        if avg > best_eval:
            best_eval = avg
            agent.save(MODEL_PATH)
            print("Saved best model to", MODEL_PATH)

    env.close()
    eval_env.close()
    print("Training complete. Best eval avg:", best_eval)

if __name__ == "__main__":
    train()
