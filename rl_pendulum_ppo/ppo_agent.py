import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from networks import ActorGaussian, Critic
import math

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.actor = ActorGaussian(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.action_low = float(action_low)
        self.action_high = float(action_high)

    @torch.no_grad()
    def act(self, state_np: np.ndarray):
        state_np = np.asarray(state_np, dtype=np.float32)

        s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu, std = self.actor(s)
        dist = Normal(mu, std)

        z = dist.sample()              # pre-squash action
        a = torch.tanh(z)              # [-1, 1]
        action = a * self.action_high  # scale to env range

        # log prob with tanh correction
        logp = dist.log_prob(z).sum(dim=1)
        correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=1)
        logp = logp - correction

        value = self.critic(s)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        return action_np, float(logp.item()), float(value.item())




    def compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self, batch, epochs=10, minibatch_size=256):
        states, actions, old_logps, returns, advantages = batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_logps_t = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n = states.shape[0]
        idxs = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                mu, std = self.actor(states_t[mb])
                dist = Normal(mu, std)

                # actions are in env scale [-2,2], convert back to [-1,1] then atanh
                a = actions_t[mb] / self.action_high
                a = torch.clamp(a, -0.999, 0.999)
                z = 0.5 * torch.log((1 + a) / (1 - a))  # atanh

                logps = dist.log_prob(z).sum(dim=1)
                correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=1)
                logps = logps - correction

                entropy = dist.entropy().sum(dim=1).mean()
    
                ratio = torch.exp(logps - old_logps_t[mb])

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(states_t[mb])
                value_loss = nn.MSELoss()(values_pred, returns_t[mb])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.opt.step()
    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.eval()
        self.critic.eval()
