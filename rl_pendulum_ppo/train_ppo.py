import os
import numpy as np
import gymnasium as gym

from ppo_agent import PPOAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppo_pendulum.pt")

def collect_rollout(env, agent, rollout_steps=4096):
    states, actions, logps, rewards, values, dones = [], [], [], [], [], []

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

    _, _, last_value = agent.act(state)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(logps, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        float(last_value),
    )

def train():
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]

    agent = PPOAgent(
        state_dim, action_dim,
        action_low=action_low, action_high=action_high,
        lr=3e-4, gamma=0.95, gae_lambda=0.95,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.0
    )

    updates = 150
    rollout_steps = 4096
    epochs = 10
    minibatch_size = 512

    best_eval = -1e9

    for u in range(updates):
        states, actions, logps, rewards, values, dones, last_value = collect_rollout(env, agent, rollout_steps)
        adv, returns = agent.compute_gae(rewards, values, dones, last_value)

        agent.update((states, actions, logps, returns.astype(np.float32), adv.astype(np.float32)),
                     epochs=epochs, minibatch_size=minibatch_size)

        # quick eval: deterministic (use mean action)
        avg = evaluate(eval_env, agent, episodes=5)
        print(f"Update {u}, eval avg return: {avg:.1f}")

        if avg > best_eval:
            best_eval = avg
            agent.save(MODEL_PATH)
            print("Saved best model to", MODEL_PATH)

    env.close()
    eval_env.close()
    print("Training complete. Best eval:", best_eval)

def evaluate(env, agent, episodes=5):
    scores = []
    for ep in range(episodes):
        state, info = env.reset(seed=1000 + ep)
        total = 0.0
        for _ in range(200):
            # deterministic: use mean action (no sampling)
            s = np.asarray(state, dtype=np.float32)
            import torch
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
                mu, _ = agent.actor(st)
                action = mu.squeeze(0).cpu().numpy()
                action = np.clip(action, agent.action_low, agent.action_high)

            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    return float(np.mean(scores))

if __name__ == "__main__":
    train()
